import numpy as np
import jax
import jax.numpy as jnp
import numpy as np
import tiktoken

from basicgpt.tiny_stories import TOKENIZER_SIZE
import basicgpt.transformer as transformer

from libgpt.corpus import InputCorpus
from libgpt.corpus import seed_corpus_from_numpy_arrays
from libgpt.fuzzer import Fuzzer
from libgpt.mutation_functions import do_basic_mutations
from libgpt.sample_functions import recent_sample_function

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
            loss = transformer.masked_cross_entropy(logits, prompt_tokens_array, mask)
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)  # has_aux=True since we're returning extra (auxiliary) data
        (loss, logits), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)

        coverage = [logits[0]]
        metadata = [logits[0]]
        return state, loss, coverage, metadata
    
    return fetch_function

if __name__ == "__main__":
    enc = tiktoken.encoding_for_model("gpt2")
    prompt = "Once upon a time, there was a little car named Beep. Beep loved to go fast and play in the sun. Beep was a healthy car because he always had good fuel. Good fuel made Beep happy and strong. One day, Beep was driving in the park when he saw a big tree. The tree had many leaves that were falling. Beep liked how the leaves fall and wanted to play with them. Beep drove under the tree and watched the leaves fall on him. He laughed and beeped his horn. Beep played with the falling leaves all day. When it was time to go home, Beep knew he needed more fuel. He went to the fuel place and got more healthy fuel. Now, Beep was ready to go fast and play again the next day. And Beep lived happily ever after."
    prompt_tokens = enc.encode(prompt)
    if len(prompt_tokens) > transformer.MAX_LEN:
        prompt_tokens = prompt_tokens[:transformer.MAX_LEN]

    prompt_length = len(prompt_tokens)
    
    # Create prompt_tokens_array using jnp.concatenate with dtype=int64
    prompt_tokens_padded = jnp.concatenate([jnp.array(prompt_tokens, dtype=jnp.int32), jnp.zeros(transformer.MAX_LEN - prompt_length, dtype=jnp.int32)])
    prompt_tokens_array = jnp.expand_dims(prompt_tokens_padded, axis=0)

    # Create the mask using jnp.concatenate with dtype=float32
    mask_padded = jnp.concatenate(
        
        [jnp.ones(prompt_length, dtype=jnp.bfloat16), jnp.zeros(transformer.MAX_LEN - prompt_length, dtype=jnp.bfloat16)])
    mask = jnp.expand_dims(mask_padded, axis=0)

    MLP_PATH = "basicgpt/checkpoints/params_MLP.npy"
    params = np.load(MLP_PATH, allow_pickle=True).item()

    print("Creating model...")
    state = transformer.create_train_state(transformer.rng, transformer.config)
    print("Number of parameters: ", transformer.param_count(state.params))
    print("Number of non-embedding parameters:", transformer.param_count(state.params) - (transformer.config["d_model"] * TOKENIZER_SIZE * 2 + transformer.config["d_model"] * transformer.MAX_LEN) / 1e6)

    fetch_function = build_fetch_function()

    state, loss, coverage, metadata = fetch_function(state, prompt_tokens_array, mask)

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

        