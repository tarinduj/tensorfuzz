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
"""Defines the actual Fuzzer object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from libgpt.corpus import CorpusElement

# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
class Fuzzer(object):
    """Class representing the fuzzer itself."""

    def __init__(
        self,
        corpus,
        coverage_function,
        metadata_function,
        objective_function,
        mutation_function,
        fetch_function,
    ):
        """Init the class.

        Args:
          corpus: An InputCorpus object.
          coverage_function: a function that does CorpusElement -> Coverage.
          metadata_function: a function that does CorpusElement -> Metadata.
          objective_function: a function that checks if a CorpusElement satisfies
            the fuzzing objective (e.g. find a NaN, find a misclassification, etc).
          mutation_function: a function that does CorpusElement -> Metadata.
          fetch_function: grabs numpy arrays from the JAX runtime using the relevant
            functions.
        Returns:
          Initialized object.
        """
        self.corpus = corpus
        self.coverage_function = coverage_function
        self.metadata_function = metadata_function
        self.objective_function = objective_function
        self.mutation_function = mutation_function
        self.fetch_function = fetch_function

    def loop(self, iterations):
        """Fuzzes a machine learning model in a loop, making *iterations* steps."""

        for iteration in range(iterations):
            if iteration % 100 == 0:
                print(f"fuzzing iteration: {iteration}")
            parent = self.corpus.sample_input()

            # Get a mutated batch for each input tensor
            state, mutated_prompt_tokens, mask = self.mutation_function(parent)

            # Grab the coverage and metadata for mutated batch from the JAX runtime.
            state, _, coverage, metadata = self.fetch_function(
                state, mutated_prompt_tokens, mask
            )

            # for i, elem in enumerate(metadata):
            #     print(f"  Batch {i} shape: {elem.shape} {type(elem)}")
            #     print(f"  Batch {i} min: {elem.min()}, max: {elem.max()}")

            # Get the coverage - one from each batch element
            mutated_coverage_list = self.coverage_function(coverage)

            # Get the metadata objects - one from each batch element
            mutated_metadata_list = self.metadata_function(metadata)

            # Check for new coverage and create new corpus elements if necessary.
            # pylint: disable=consider-using-enumerate
            
            new_element = CorpusElement(
                (state, mutated_prompt_tokens, mask),
                mutated_metadata_list[0],
                mutated_coverage_list[0],
                parent,
            )
            if self.objective_function(new_element):
                return new_element
            self.corpus.maybe_add_to_corpus(new_element)

        return None
