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

def print_structure(obj, indent=0):
    """
    Recursively print the type structure of a data structure.
    
    Args:
    obj: The object to inspect
    indent: The current indentation level (used for recursive calls)
    
    Returns:
    None (prints to stdout)
    """
    indent_str = "  " * indent
    obj_type = type(obj).__name__

    if isinstance(obj, (dict, jax.tree_util.PyTreeDef)):
        print(f"{indent_str}{obj_type}:")
        if isinstance(obj, dict):
            for key, value in obj.items():
                print(f"{indent_str}  {key}:")
                print_structure(value, indent + 2)
        else:  # PyTreeDef
            for key in obj.keys():
                print(f"{indent_str}  {key}:")
                print_structure(getattr(obj, key), indent + 2)
    elif isinstance(obj, (list, tuple)):
        print(f"{indent_str}{obj_type} of length {len(obj)}:")
        if len(obj) > 0:
            print_structure(obj[0], indent + 1)
        if len(obj) > 1:
            print(f"{indent_str}  ...")
    elif isinstance(obj, (np.ndarray, jnp.ndarray)):
        print(f"{indent_str}{obj_type} shape: {obj.shape}, dtype: {obj.dtype}")
    else:
        print(f"{indent_str}{obj_type}")

def metadata_function(metadata_batches):
    """Gets the metadata."""
    # print("metadata_batches:")
    # print_structure(metadata_batches)
    metadata_list = [
        [metadata_batches[i].ravel() for i in range(len(metadata_batches))]
    ]
    # print("metadata_list:")
    # print_structure(metadata_list)
    # exit(0)
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
    
    # # Sum up all absolute values in the entire sequence
    # total_sum = np.sum(np.abs(coverage_batch))
    
    # # Create a list with a single element
    # coverage_list = [np.array([total_sum])]

     # Calculate the sum of absolute values for each row
    coverage_array = np.sum(np.abs(coverage_batch), axis=1)
    
    coverage_list = [coverage_array]
    
    return coverage_list

def flatten_params(params):
    """Flatten a nested dictionary of parameters into a single 2D array with shape (n, 1)."""
    flat_params, tree_def = jax.tree_util.tree_flatten(params)
    concatenated = np.concatenate([p.ravel() for p in flat_params])
    return concatenated.reshape(-1, 1), tree_def

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

        