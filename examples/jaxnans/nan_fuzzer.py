# """Fuzz a neural network to get a NaN using pure JAX."""

# from typing import Dict, Tuple
# import os
# import random
# import jax
# import jax.numpy as jnp
# from jax import random as jrandom
# import numpy as np
# import optax
# from absl import flags, app

# import libjax.dataset as mnist
# from libjax import fuzz_utils
# from libjax.corpus import InputCorpus
# from libjax.corpus import seed_corpus_from_numpy_arrays
# from libjax.coverage_functions import all_logit_coverage_function
# from libjax.fuzzer import Fuzzer
# from libjax.mutation_functions import do_basic_mutations
# from libjax.sample_functions import recent_sample_function

# FLAGS = flags.FLAGS
# flags.DEFINE_string(
#     "checkpoint_dir", "/tmp/jaxnanfuzzer", "Dir containing checkpoints of model to fuzz."
# )
# flags.DEFINE_integer(
#     "total_inputs_to_fuzz", 100, "Loops over the whole corpus."
# )
# flags.DEFINE_integer(
#     "mutations_per_corpus_item", 100, "Number of times to mutate corpus item."
# )
# flags.DEFINE_float(
#     "ann_threshold",
#     1.0,
#     "Distance below which we consider something new coverage.",
# )
# flags.DEFINE_integer("seed", None, "Random seed for both python and numpy.")
# flags.DEFINE_boolean(
#     "random_seed_corpus", False, "Whether to choose a random seed corpus."
# )

# # Define layer functions
# def dense(params: Dict, x: jnp.ndarray, out_dim: int, key: str) -> jnp.ndarray:
#     w, b = params[f'{key}_w'], params[f'{key}_b']
#     return jnp.dot(x, w) + b

# def relu(x: jnp.ndarray) -> jnp.ndarray:
#     return jnp.maximum(0, x)

# # Define model function
# def classifier(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
#     x = x.reshape((x.shape[0], -1))  # flatten
#     x = dense(params, x, 200, 'layer1')
#     x = relu(x)
#     x = dense(params, x, 100, 'layer2')
#     x = relu(x)
#     x = dense(params, x, 10, 'layer3')
#     return x

# def unsafe_softmax(logits: jnp.ndarray) -> jnp.ndarray:
#     return jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=1, keepdims=True)

# def unsafe_cross_entropy(probabilities: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
#     return -jnp.sum(labels * jnp.log(probabilities), axis=1)

# @jax.jit
# def model_fn(params: Dict, images: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     logits = classifier(params, images)
#     bad_softmax = unsafe_softmax(logits)
#     return logits, bad_softmax

# def metadata_function(metadata_batches):
#     """Gets the metadata."""
#     metadata_list = [
#         [metadata_batches[i][j] for i in range(len(metadata_batches))]
#         for j in range(metadata_batches[0].shape[0])
#     ]
#     return metadata_list

# def objective_function(corpus_element):
#     """Checks if the metadata is inf or NaN."""
#     metadata = corpus_element.metadata
#     if all([np.isfinite(d).all() for d in metadata]):
#         return False

#     print("Objective function satisfied: non-finite element found.")
#     return True

# def convert_params(params_dict):
#     """Convert loaded parameters to JAX arrays."""
#     converted_params = {}
#     for k, v in params_dict.items():
#         if isinstance(v, np.ndarray):
#             converted_params[k] = jnp.array(v)
#         elif isinstance(v, dict):
#             converted_params[k] = convert_params(v)
#         else:
#             # For scalar values or other types
#             converted_params[k] = jnp.array(v)
#     return converted_params

# def main(_):
#     # Set the seeds
#     if FLAGS.seed:
#         random.seed(FLAGS.seed)
#         np.random.seed(FLAGS.seed)

#     coverage_function = all_logit_coverage_function
#     image, label = fuzz_utils.basic_mnist_input_corpus(
#         choose_randomly=FLAGS.random_seed_corpus
#     )

#     print("image type: ", type(image), image.shape)
#     print("label type: ", type(label), label.shape)

#     exit(0)

#     numpy_arrays = [[image, label]]

#     # Load the model
#     checkpoint_file = os.path.join(FLAGS.checkpoint_dir, "checkpoint_34000.npz")
#     with np.load(checkpoint_file, allow_pickle=True) as data:
#         params = data['params'].item()  # Assuming params were saved under 'params' key
#         params = convert_params(params)

#     # Create a fetch function
#     def fetch_function(input_batches):
#         images = jnp.array(input_batches[0])
#         logits, bad_softmax = model_fn(params, images)
#         return [logits], [bad_softmax, logits]

#     size = FLAGS.mutations_per_corpus_item
#     mutation_function = lambda elt: do_basic_mutations(elt, size)
#     seed_corpus = seed_corpus_from_numpy_arrays(
#         numpy_arrays, coverage_function, metadata_function, fetch_function
#     )
#     corpus = InputCorpus(
#         seed_corpus, recent_sample_function, FLAGS.ann_threshold, "kdtree"
#     )
#     fuzzer = Fuzzer(
#         corpus,
#         coverage_function,
#         metadata_function,
#         objective_function,
#         mutation_function,
#         fetch_function,
#     )
#     result = fuzzer.loop(FLAGS.total_inputs_to_fuzz)
#     if result is not None:
#         print("Fuzzing succeeded.")
#         print(
#             f"Generations to make satisfying element: {result.oldest_ancestor()[1]}."
#         )
#     else:
#         print("Fuzzing failed to satisfy objective function.")

# if __name__ == "__main__":
#     app.run(main)

import random
import numpy as np
import os
from absl import flags, app

from libjax import fuzz_utils
from libjax.corpus import InputCorpus
from libjax.corpus import seed_corpus_from_numpy_arrays
from libjax.coverage_functions import all_logit_coverage_function
from libjax.fuzzer import Fuzzer
from libjax.mutation_functions import do_basic_mutations
from libjax.sample_functions import recent_sample_function

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_dir", None, "Dir containing checkpoints of model to fuzz.")
flags.DEFINE_integer("total_inputs_to_fuzz", 100, "Loops over the whole corpus.")
flags.DEFINE_integer("mutations_per_corpus_item", 100, "Number of times to mutate corpus item.")
flags.DEFINE_float("ann_threshold", 1.0, "Distance below which we consider something new coverage.")
flags.DEFINE_integer("seed", None, "Random seed for both python and numpy.")
flags.DEFINE_boolean("random_seed_corpus", False, "Whether to choose a random seed corpus.")

def metadata_function(metadata_batches):
    """Gets the metadata."""
    metadata_list = [
        [metadata_batches[i][j] for i in range(len(metadata_batches))]
        for j in range(metadata_batches[0].shape[0])
    ]
    return metadata_list

def objective_function(corpus_element):
    """Checks if the metadata is inf or NaN."""
    metadata = corpus_element.metadata
    if all([np.isfinite(d).all() for d in metadata]):
        return False

    print("Objective function satisfied: non-finite element found.")
    return True

def main(_):
    """Constructs the fuzzer and performs fuzzing."""
    if FLAGS.seed:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    coverage_function = all_logit_coverage_function
    image, label = fuzz_utils.basic_mnist_input_corpus(choose_randomly=FLAGS.random_seed_corpus)
    
    print("image type: ", type(image), image.shape)
    print("label type: ", type(label), label.shape)
    
    numpy_arrays = [[image, label]]

    # Get the latest checkpoint
    checkpoint_files = [f for f in os.listdir(FLAGS.checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".npz")]
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, latest_checkpoint)

    params, tensor_map = fuzz_utils.get_tensors_from_checkpoint(checkpoint_path)
    fetch_function = fuzz_utils.build_fetch_function(params, tensor_map)

    # Test the fetch function
    print("Testing fetch function...")
    test_image = np.ones((2, 28, 28, 1))  # Create a dummy image batch of all ones with batch size 2
    test_label = np.array([5, 3])
    test_input_batches = [test_image, test_label]

    coverage_batches, metadata_batches = fetch_function(test_input_batches)

    print("Coverage batches:", type(coverage_batches))
    for i, batch in enumerate(coverage_batches):
        print(f"  Batch {i} shape: {batch.shape} {type(batch)}")
        print(f"  Batch {i} min: {batch.min()}, max: {batch.max()}")

    print("Metadata batches:", type(metadata_batches))
    for i, batch in enumerate(metadata_batches):
        print(f"  Batch {i} shape: {batch.shape} {type(batch)}")
        print(f"  Batch {i} min: {batch.min()}, max: {batch.max()}")

    size = FLAGS.mutations_per_corpus_item
    mutation_function = lambda elt: do_basic_mutations(elt, size)
    seed_corpus = seed_corpus_from_numpy_arrays(
        numpy_arrays, coverage_function, metadata_function, fetch_function
    )
    corpus = InputCorpus(
        seed_corpus, recent_sample_function, FLAGS.ann_threshold, "kdtree"
    )
    fuzzer = Fuzzer(
        corpus,
        coverage_function,
        metadata_function,
        objective_function,
        mutation_function,
        fetch_function,
    )
    result = fuzzer.loop(FLAGS.total_inputs_to_fuzz)
    if result is not None:
        print("Fuzzing succeeded.")
        print(
            f"Generations to make satisfying element: {result.oldest_ancestor()[1]}."
        )
    else:
        print("Fuzzing failed to satisfy objective function.")

if __name__ == "__main__":
    app.run(main)