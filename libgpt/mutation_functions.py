# Copyright 2018 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions that mutate inputs for coverage guided fuzzing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


# pylint: disable=too-many-locals
def do_basic_mutations(
    corpus_element, mutations_count, a_min=0, a_max=50256
):
    """Mutates image inputs with white noise.

  Args:
    corpus_element: A CorpusElement object. It's assumed in this case that
      corpus_element.data[0] is a numpy representation of an image and
      corput_element.data[1] is a label or something we *don't* want to change.
    mutations_count: Integer representing number of mutations to do in
      parallel.
    constraint: If not None, a constraint on the norm of the total mutation.

  Returns:
    A list of batches, the first of which is mutated images and the second of
    which is passed through the function unchanged (because they are image
    labels or something that we never intended to mutate).
  """
    # Here we assume the corpus.data is of the form (image, label)
    # We never mutate the label.
    state, prompt_tokens, mask = corpus_element.data

    noise = np.random.randint(-a_max/2, a_max/2 + 1, size=prompt_tokens.shape)

    # print(f"prompt_tokens  min: {prompt_tokens.min()}, max: {prompt_tokens.max()}")

    mutated_prompt_tokens = prompt_tokens + noise
    mutated_prompt_tokens = np.abs(mutated_prompt_tokens)
    mutated_prompt_tokens = np.clip(
        mutated_prompt_tokens, a_min=a_min, a_max=a_max
    )

    # print(f"mutated_prompt_tokens  min: {mutated_prompt_tokens.min()}, max: {mutated_prompt_tokens.max()}")
    
    return state, mutated_prompt_tokens, mask
