"""Train a model that is likely to have NaNs using pure JAX."""

from typing import Dict, Tuple, Callable
import os
import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import optax
import numpy as np
import numpy.random as npr

from absl import flags, app

import libjax.dataset as mnist

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "checkpoint_dir",
    "/tmp/jaxnanfuzzer",
    "The overall dir in which we store experiments",
)
flags.DEFINE_string(
    "data_dir", "/tmp/mnist", "The directory in which we store the MNIST data"
)
flags.DEFINE_integer(
    "training_steps", 35000, "Number of mini-batch gradient updates to perform"
)
flags.DEFINE_float(
    "init_scale", 0.25, "Scale of weight initialization for classifier"
)

# Define layer functions
def dense(params: Dict, x: jnp.ndarray, out_dim: int, key: str) -> jnp.ndarray:
    w, b = params[f'{key}_w'], params[f'{key}_b']
    return jnp.dot(x, w) + b

def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(0, x)

# Define model function
def classifier(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
    x = x.reshape((x.shape[0], -1))  # flatten
    x = dense(params, x, 200, 'layer1')
    x = relu(x)
    x = dense(params, x, 100, 'layer2')
    x = relu(x)
    x = dense(params, x, 10, 'layer3')
    return x

# Initialize parameters
def init_params(rng: jnp.ndarray, input_shape: Tuple[int, ...]) -> Dict:
    def init_layer(key, in_dim, out_dim):
        k1, k2 = random.split(key)
        w = random.uniform(k1, (in_dim, out_dim), minval=-FLAGS.init_scale, maxval=FLAGS.init_scale)
        b = random.uniform(k2, (out_dim,), minval=-FLAGS.init_scale, maxval=FLAGS.init_scale)
        return w, b

    rng1, rng2, rng3 = random.split(rng, 3)
    flat_dim = int(np.prod(input_shape[1:]))
    return {
        'layer1_w': init_layer(rng1, flat_dim, 200)[0],
        'layer1_b': init_layer(rng1, flat_dim, 200)[1],
        'layer2_w': init_layer(rng2, 200, 100)[0],
        'layer2_b': init_layer(rng2, 200, 100)[1],
        'layer3_w': init_layer(rng3, 100, 10)[0],
        'layer3_b': init_layer(rng3, 100, 10)[1],
    }

# Define unsafe operations
def unsafe_softmax(logits: jnp.ndarray) -> jnp.ndarray:
    """Computes softmax in a numerically unstable way."""
    return jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=1, keepdims=True)

def unsafe_cross_entropy(probabilities: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Computes cross entropy in a numerically unstable way."""
    return -jnp.sum(labels * jnp.log(probabilities), axis=1)

# Define loss function
def loss_fn(params: Dict, images: jnp.ndarray, labels: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple]:
    logits = classifier(params, images)
    print("logits shape: ", logits.shape)
    bad_softmax = unsafe_softmax(logits)
    print("bad_softmax shape: ", bad_softmax.shape)
    print("labels shape: ", labels.shape)
    bad_cross_entropies = unsafe_cross_entropy(bad_softmax, labels)
    loss = jnp.mean(bad_cross_entropies)
    return loss, (logits, bad_softmax, bad_cross_entropies)

# Training step
@jit
def train_step(params: Dict, opt_state: optax.OptState, images: jnp.ndarray, labels: jnp.ndarray) -> Tuple[Dict, optax.OptState, jnp.ndarray, Tuple]:
    (loss, aux), grads = value_and_grad(loss_fn, argnums=0, has_aux=True)(params, images, labels)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# Accuracy computation
@jit
def compute_accuracy(params: Dict, images: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    logits = classifier(params, images)
    return jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))

def main(_):
    # Initialize random key
    print(FLAGS.training_steps)
    rng = random.PRNGKey(0)

    # Load and prepare dataset
    batch_size = 100
    images, labels  = mnist.train(FLAGS.data_dir)
    num_train = images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield images[batch_idx], labels[batch_idx]
    
    batches = data_stream()

    # Initialize model and optimizer
    rng, init_rng = random.split(rng)
    params = init_params(init_rng, (1, 28, 28, 1))
    global optimizer  # Make optimizer global so it's accessible in train_step
    learning_rate = 0.01
    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(params)

     # Training loop
    for step in range(FLAGS.training_steps):
        step_images, step_labels = next(batches)
        step_images = jnp.array(step_images).reshape((-1, 28, 28, 1))
        step_labels = jnp.array(step_labels).reshape(-1)
        step_labels_one_hot = jax.nn.one_hot(jnp.array(step_labels), 10)

        params, opt_state, loss, (logits, bad_softmax, bad_cross_entropies) = train_step(params, opt_state, step_images, step_labels_one_hot)

        if step % 1000 == 0:
            accuracy = compute_accuracy(params, step_images, step_labels_one_hot)
            print(f"step: {step}, loss: {loss}, accuracy: {accuracy}")

            # Save checkpoint with additional information
            os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)
            checkpoint_data = {
                'params': params,
                'input_tensors': {'images': step_images.shape, 'labels': step_labels.shape},
                'coverage_tensors': {'logits': logits.shape},
                'metadata_tensors': {
                    'bad_softmax': bad_softmax.shape,
                    'bad_cross_entropies': bad_cross_entropies.shape,
                    'logits': logits.shape
                }
            }
            with open(os.path.join(FLAGS.checkpoint_dir, f"checkpoint_{step}.npz"), 'wb') as f:
                np.savez(f, **checkpoint_data)

    print("Training completed.")

if __name__ == "__main__":
    app.run(main)