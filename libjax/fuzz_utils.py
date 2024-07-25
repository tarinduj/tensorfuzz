# Copyright 2018 Google LLC (Modified for JAX)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for the JAX-based fuzzer library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from typing import Any, Callable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import libjax.dataset as mnist

# Type aliases
Array = Any  # jax.Array or np.ndarray
Model = Callable[[Array], Tuple[Array, Array]]  # (input) -> (coverage, metadata)
PRNGKey = Any


def basic_mnist_input_corpus(choose_randomly: bool = False, data_dir: str = "/tmp/mnist"):
    """Returns an image and label from MNIST.

    Args:
      choose_randomly: Whether to choose a random image or the first one.
      data_dir: The directory containing the MNIST data.

    Returns:
      A tuple containing a single image and its label.
    """
    images, labels = mnist.train(data_dir)
    
    if choose_randomly:
        idx = random.randint(0, images.shape[0] - 1)
    else:
        idx = 0

     # Convert JAX array to numpy array and reshape
    image = np.array(images[idx]).reshape(28, 28, 1).astype(np.float32)
    
    # Convert JAX array to numpy scalar
    label = np.int32(labels[idx].item())

    print(f"Seeding corpus with element at idx: {idx}")
    return image, label


def get_tensors_from_checkpoint(checkpoint_path):
    """Loads and returns the fuzzing tensors from a JAX checkpoint."""
    with np.load(checkpoint_path, allow_pickle=True) as data:
        params = data['params'].item()
        input_tensors = data['input_tensors'].item()
        coverage_tensors = data['coverage_tensors'].item()
        metadata_tensors = data['metadata_tensors'].item()
    
    tensor_map = {
        "input": input_tensors,
        "coverage": coverage_tensors,
        "metadata": metadata_tensors,
    }
    return params, tensor_map

def build_fetch_function(params, tensor_map):
    """Constructs fetch function for JAX model."""
    def fetch_function(input_batches):
        images = jnp.array(input_batches[0])
        logits = classifier(params, images)
        bad_softmax = unsafe_softmax(logits)
        bad_cross_entropies = unsafe_cross_entropy(bad_softmax, jax.nn.one_hot(input_batches[1], 10))
        
        # Convert JAX arrays to NumPy arrays
        logits_np = np.array(logits)
        bad_softmax_np = np.array(bad_softmax)
        bad_cross_entropies_np = np.array(bad_cross_entropies)
        
        coverage_batches = [logits_np]
        metadata_batches = [bad_softmax_np, bad_cross_entropies_np, logits_np]
        
        print("input batches shape: ", len(input_batches), input_batches[0].shape)
        print("coverage batches shape: ", len(coverage_batches), coverage_batches[0].shape)
        print("metadata batches shape: ", len(metadata_batches), metadata_batches[0].shape)
    
        return coverage_batches, metadata_batches
    
    return fetch_function

#####

from typing import Dict

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

def unsafe_softmax(logits: jnp.ndarray) -> jnp.ndarray:
    """Computes softmax in a numerically unstable way."""
    return jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=1, keepdims=True)

def unsafe_cross_entropy(probabilities: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Computes cross entropy in a numerically unstable way."""
    return -jnp.sum(labels * jnp.log(probabilities), axis=1)