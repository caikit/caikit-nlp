# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Huggingface auto causal LM resource type
"""
# Standard
from collections.abc import Mapping
from copy import deepcopy
from typing import Union

# Third Party
from transformers import (
    AutoModelForCausalLM,
    BatchEncoding,
    DataCollatorForLanguageModeling,
)
from transformers.models.auto import modeling_auto

# First Party
from caikit.core.data_model import DataStream
from caikit.core.modules import module

# Local
from ...data_model import GenerationTrainRecord, PromptOutputModelType
from ...toolkit.verbalizer_utils import render_verbalizer
from .base import PretrainedModelBase


@module(
    id="6759e891-287b-405b-bd8b-54a4a4d51c24",
    name="HF Transformers Auto Causal LM",
    version="0.1.0",
)
class HFAutoCausalLM(PretrainedModelBase):
    """This resource (module) wraps a handle to a Huggingface AutoModelForCausalLM"""

    MODEL_TYPE = AutoModelForCausalLM
    SUPPORTED_MODEL_TYPES = modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    TASK_TYPE = "CAUSAL_LM"
    PROMPT_OUTPUT_TYPES = [PromptOutputModelType.DECODER]
    MAX_NUM_TRANSFORMERS = 1
    REQUIRES_TOKEN_UNWRAPPING = True

    @classmethod
    def tokenize_function(
        cls,
        example: Union[GenerationTrainRecord, Mapping],
        tokenizer: "AutoTokenizer",
        max_source_length: int,
        max_target_length: int,
        verbalizer: Union[None, str] = None,
        task_ids: Union[None, int] = None,
    ) -> DataStream["BatchEncoding"]:
        """Tokenization function to be used for causallm training; this function consumes a
        GenerationTrainRecord object and applies the verbalizer to it followed by
        the model tokenizer. Due to the nature of our training data with src/target seqs,
        each sample yields one example per token in the target sequence.

        Args:
            example: GenerationTrainRecord | Mapping
                Training data model object to convert a form we can learn on, or a Mapping
                that has keys input/output.

        Returns:
            DataStream[transformers.tokenization_utils_base.BatchEncoding]
                stream of encoded tokenization output corresponding to the input example.
        """
        # Extract the source & target from our provided inputs
        source, target = cls.decompose_example_io(example)
        # Determine if our mapped inputs are in batched mode or not
        batched_mode = isinstance(source, list) and isinstance(target, list)

        # TODO: Handle batched verbalizer stuff!
        if batched_mode and verbalizer is not None:
            raise NotImplementedError(
                "Verbalizer rendering not implemented for batch mode"
            )
        source = (
            source if verbalizer is None else render_verbalizer(verbalizer, example)
        )

        # HACK: We shouldn't have to pad here, but the causal LM data collator dynamic padding
        # does not appear to be playing nicely with the Huggingface trainer / torch fsdp...
        source_ids = tokenizer(source, max_length=max_source_length, truncation=True)
        target_ids = tokenizer(target, max_length=max_target_length, truncation=True)
        if batched_mode:
            num_target_samples = []
            for idx, _ in enumerate(source_ids.input_ids):
                source_ids["input_ids"][idx] = (
                    source_ids.input_ids[idx] + target_ids.input_ids[idx]
                )
                num_target_samples.append(len(target_ids.input_ids[idx]))
                if task_ids is not None:
                    source_ids["task_ids"][idx] = task_ids
        else:
            source_ids["input_ids"] = source_ids.input_ids + target_ids.input_ids
            # Here, we need to yield and manipulate the attention mask to attend
            # to the input seq + the tokens we have seen so far...
            num_target_samples = len(target_ids.input_ids)

            if task_ids is not None:
                source_ids["task_ids"] = task_ids

        # This is disgusting! TODO:
        # - Consolidate batched [generator] vs. non-batched behavior [batch encoded lists]
        # - Make all attention mask logic common, etc.
        def single_generator_func():
            for idx in range(num_target_samples):
                # This may not actually be needed, but for now we do it, since the underlying
                # data may be referenced in multiple places, and the data will be dynamically
                # padded by the LM collator
                s = deepcopy(source_ids)
                s["attention_mask"] = (
                    s["attention_mask"]
                    + [1] * (idx + 1)
                    + [0] * (num_target_samples - idx - 1)
                )
                yield s

        def get_batched_output():
            # Initialize the batch encoding key lists as empty
            batch_encoding = BatchEncoding()
            for k in source_ids:
                batch_encoding[k] = []
            # Flatten the batch and add everything individually...
            for batch_idx in range(len(source_ids.input_ids)):
                # Consider every output text for this entry in the batch
                for idx in range(num_target_samples[batch_idx]):
                    # Create the batch encoding dict directly and populate the keys
                    # from the corresponding entry inside of the batch...
                    for key in source_ids:
                        if key != "attention_mask":
                            batch_encoding[key].append(source_ids[key][batch_idx])
                        else:
                            # Handle the attention mask for this entry...
                            attn_mask = (
                                source_ids["attention_mask"][batch_idx]
                                + [1] * (idx + 1)
                                + [0] * (num_target_samples[batch_idx] - idx - 1)
                            )
                            batch_encoding["attention_mask"].append(attn_mask)
            return batch_encoding

        if batched_mode:
            return get_batched_output()
        return DataStream(single_generator_func)

    def _get_data_collator(self, **kwargs):
        """Function to return appropriate data collator based on resource.

        DataCollatorForLanguageModeling is used here which will dynamically
        padded to maximum length of a batch if they are not all of the same
        length.

        NOTE: If mlm (masked language modeling) is not passed in kwargs,
        this function will automatically set it to `False`.

        Args:
            **kwargs:
                All the keyword arguments passed to this function
                will get filtered out to appropriate ones that are
                applicable to implemented data collator.
        Returns:
            transformers.DataCollator
        """

        applicable_args = ["mlm", "pad_to_multiple_of"]
        collator_kwargs = {key: kwargs[key] for key in applicable_args if key in kwargs}

        if "mlm" not in collator_kwargs:
            collator_kwargs["mlm"] = False

        return DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer, return_tensors="pt", **collator_kwargs
        )
