import numpy as np
from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Literal, TypedDict
import numpy as np
from tinygpt.tiny_stories import TinyStoriesDataset
from tinygpt.tiny_stories import TOKENIZER_SIZE
from flax.training import train_state
import optax
from tinygpt.chebykan_layer import ChebyKAN
from time import perf_counter
import tiktoken

from libgpt.corpus import InputCorpus
from libgpt.corpus import seed_corpus_from_numpy_arrays
from libgpt.fuzzer import Fuzzer
from libgpt.mutation_functions import do_basic_mutations
from libgpt.sample_functions import recent_sample_function

D_TYPE = jnp.float16

MAX_LEN = 64

class MLP(nn.Module):

    @nn.compact
    def __call__(self, x):
        d_outer = x.shape[-1]
        # 84 is choosen so that the number of parameters matches then KAN layer
        x = nn.Dense(features=768, param_dtype=D_TYPE)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=d_outer, param_dtype=D_TYPE)(x)
        return x
    
class MLPBlock(nn.Module):

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(param_dtype=D_TYPE)(x)
        y = MLP()(y)
        return x + y


class SelfAttentionBlock(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x):
        # Shape (batch_size, seq_len, d_model)
        n_heads, d_model = self.n_heads, self.d_model
        assert d_model % n_heads == 0, 'n_heads must divide d_model'
        # Shape (batch_size, num_heads, seq_len, seq_len)
        mask = jnp.ones((x.shape[0], n_heads, x.shape[1], x.shape[1]))
        # Create diagonal mask
        mask = jnp.tril(mask)
        y = nn.LayerNorm(param_dtype=D_TYPE)(x)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=n_heads, qkv_features=d_model // n_heads, out_features=d_model, param_dtype=D_TYPE
        )(y, mask=mask)
        return x + attn

class Transformer(nn.Module):
    d_model: int
    n_heads: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        # Shape (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = nn.Embed(num_embeddings=TOKENIZER_SIZE, features=self.d_model, param_dtype=D_TYPE)(x)
        pos_emb = nn.Embed(num_embeddings=MAX_LEN, features=self.d_model, param_dtype=D_TYPE)(jnp.arange(MAX_LEN))
        x = x + pos_emb
        for _ in range(self.n_layers):
            x = SelfAttentionBlock(self.d_model, self.n_heads)(x)
            x = MLPBlock()(x)
        # Shape (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        x = nn.Dense(features=TOKENIZER_SIZE, use_bias=False, param_dtype=D_TYPE, dtype=D_TYPE)(x)
        return x

def param_count(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6

class Config(TypedDict):
    d_model: int
    n_heads: int
    n_layers: int
    learning_rate: float
    max_steps: int
    batch_size: int
    weight_decay: float
    block_type: Literal["MLP", "KAN", "Hybrid"]

def unsafe_softmax(logits: jnp.ndarray) -> jnp.ndarray:
    """Computes softmax in a numerically unstable way."""
    return jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=-1, keepdims=True)

def masked_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray, mask: jnp.ndarray):
    # logits shape: (batch_size, seq_len, vocab_size)
    # targets shape: (batch_size, seq_len)
    # mask shape: (batch_size, seq_len)
    # shift everything by 1
    logits = logits[:, :-1, :]
    targets = targets[:, 1:]
    mask = mask[:, 1:]
    vocab_size = logits.shape[-1]
    one_hot_targets = jax.nn.one_hot(targets, vocab_size)
    # one_hot_targets shape: (batch_size, seq_len, vocab_size)
    probs = jax.nn.softmax(logits)
    # probs = unsafe_softmax(logits)
    log_probs = jnp.log(probs)
    loss = -jnp.sum(log_probs * one_hot_targets, axis=-1)
    # loss shape: (batch_size, seq_len)
    loss = loss * mask
    # Flatten everything divide by the sum of the mask
    total_tokens = jnp.sum(mask.flatten())
    return jnp.sum(loss.flatten()) / total_tokens, probs
    


def create_train_state(params, config):
    if config["block_type"] == "MLP":
        model = Transformer(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
        )
   
    optimizer = optax.adamw(learning_rate=config['learning_rate'], weight_decay=config['weight_decay'])
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

rng = jax.random.PRNGKey(0)

config = Config(
    d_model=128,
    n_heads=8,
    n_layers=1,
    learning_rate=1e-5,
    batch_size=1,
    weight_decay=0.001,
    block_type="MLP", 
)

def objective_function(corpus_element):
    """Checks if the metadata is inf or NaN."""
    metadata = corpus_element.metadata
    if all([np.isfinite(d).all() for d in metadata]):
        return False

    print("Objective function satisfied: non-finite element found.")
    return True

def metadata_function(metadata_batches):
    """Gets the metadata."""
    metadata_list = [
        [metadata_batches[i][j] for i in range(len(metadata_batches))]
        for j in range(metadata_batches[0].shape[0])
    ]
    return metadata_list

def coverage_function(coverage_batches):
    """Computes coverage based on the sum of the absolute values of the logits.

    Args:
        coverage_batches: Numpy arrays containing coverage information pulled from
          a call to sess.run. In this case, we assume that these correspond to a
          batch of logits.

    Returns:
        A python integer corresponding to the sum of the absolute values of the
        logits.
    """
    coverage_batch = coverage_batches[0]
    coverage_list = []
    for idx in range(coverage_batch.shape[0]):
        elt = coverage_batch[idx]
        elt = np.expand_dims(np.sum(np.abs(elt)), 0)
        coverage_list.append(elt)
    return coverage_list

def build_fetch_function():
    """Constructs fetch function (same as a train step)"""
    def fetch_function(state, prompt_tokens_array, mask):
        def loss_fn(params):
            logits = state.apply_fn(params, prompt_tokens_array)
            loss, bad_softmax = masked_cross_entropy(logits, prompt_tokens_array, mask)
            return loss, (logits, bad_softmax)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)  # has_aux=True since we're returning extra (auxiliary) data
        (loss, (logits, bad_softmax)), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)

        coverage = [logits[0]]
        metadata = [logits[0], bad_softmax[0]]
        return state, loss, coverage, metadata
    
    return fetch_function

if __name__ == "__main__":
    enc = tiktoken.encoding_for_model("gpt2")
    prompt = "Once upon a time, there was a little car named Beep. Beep loved to go fast and play in the sun. Beep was a healthy car because he always had good fuel. Good fuel made Beep happy and strong. One day, Beep was driving in the park when he saw a big tree. The tree had many leaves that were falling. Beep liked how the leaves fall and wanted to play with them. Beep drove under the tree and watched the leaves fall on him. He laughed and beeped his horn. Beep played with the falling leaves all day. When it was time to go home, Beep knew he needed more fuel. He went to the fuel place and got more healthy fuel. Now, Beep was ready to go fast and play again the next day. And Beep lived happily ever after."
    prompt_tokens = enc.encode(prompt)
    if len(prompt_tokens) > MAX_LEN:
        prompt_tokens = prompt_tokens[:MAX_LEN]

    prompt_length = len(prompt_tokens)
    
    # Create prompt_tokens_array using jnp.concatenate with dtype=int64
    prompt_tokens_padded = jnp.concatenate([jnp.array(prompt_tokens, dtype=jnp.int32), jnp.zeros(MAX_LEN - prompt_length, dtype=jnp.int32)])
    prompt_tokens_array = jnp.expand_dims(prompt_tokens_padded, axis=0)

    # Create the mask using jnp.concatenate with dtype=float32
    mask_padded = jnp.concatenate(
        
        [jnp.ones(prompt_length, dtype=jnp.bfloat16), jnp.zeros(MAX_LEN - prompt_length, dtype=jnp.bfloat16)])
    mask = jnp.expand_dims(mask_padded, axis=0)

    # Both prompt_tokens_array and mask should now have shape (1, MAX_LEN)
    # print("prompt_tokens_array shape:", prompt_tokens_array.dtype)
    # print(prompt_tokens_array)
    # print("mask shape:", mask.dtype)
    # print(mask)

    MLP_PATH = "tinygpt/checkpoints/params_MLP.npy"
    params = np.load(MLP_PATH, allow_pickle=True).item()

    print(params['params'].keys())

    print("Creating model...")
    state = create_train_state(params, config)
    print("Number of parameters: ", param_count(state.params))
    print("Number of non-embedding parameters:", param_count(state.params) - (config["d_model"] * TOKENIZER_SIZE * 2 + config["d_model"] * MAX_LEN) / 1e6)

    fetch_function = build_fetch_function()

    state, loss, coverage, metadata = fetch_function(state, prompt_tokens_array, mask)

    # for i, elem in enumerate(metadata):
    #     print(f"  Batch {i} shape: {elem.shape} {type(elem)}")
    #     print(f"  Batch {i} min: {elem.min()}, max: {elem.max()}")
    
    # # cov_fun = coverage_function(coverage)
    # # met_fun = metadata_function(metadata)

    # print("cov fun")
    # for i, batch in enumerate( coverage_function(coverage)):
    #     print(f"  Batch {i} shape: {batch.shape} {type(batch)}")
    #     print(f"  Batch {i} min: {batch.min()}, max: {batch.max()}")

    # print("metadat fn")
    # for i, batch in enumerate(metadata_function(metadata)):
    #     print(f"  Batch {i} shape:  {type(batch)}")

    mutations_per_corpus_item = 100
    size = mutations_per_corpus_item
    mutation_function = lambda elt: do_basic_mutations(elt, size)

    seed_corpus = seed_corpus_from_numpy_arrays(
        state, prompt_tokens_array, mask, coverage_function, metadata_function, fetch_function
    )

    ann_threshold = 1.0

    corpus = InputCorpus(
        seed_corpus, recent_sample_function, ann_threshold, "kdtree"
    )

    total_inputs_to_fuzz = 1000000

    fuzzer = Fuzzer(
        corpus,
        coverage_function,
        metadata_function,
        objective_function,
        mutation_function,
        fetch_function,
    )
    result = fuzzer.loop(total_inputs_to_fuzz)

    if result is not None:
        print("Fuzzing succeeded.")
        print(
            f"Generations to make satisfying element: {result.oldest_ancestor()[1]}."
        )
    else:
        print("Fuzzing failed to satisfy objective function.")

        