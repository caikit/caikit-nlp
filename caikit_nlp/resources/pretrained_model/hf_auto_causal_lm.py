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
from copy import deepcopy
from collections.abc import Mapping
from typing import Callable, Tuple, Union

# Third Party
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
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
        # Only render the verbalizer if one is provided
        source, target = cls.decompose_example_io(example)
        source = source if verbalizer is None else render_verbalizer(verbalizer, example)

        source_ids = tokenizer(
            source, max_length=max_source_length, truncation=True
        )
        target_ids = tokenizer(
            target, max_length=max_target_length, truncation=True
        )
        source_ids["input_ids"] = source_ids.input_ids + target_ids.input_ids
        # Here, we need to yield and manipulate the attention mask to attend
        # to the input seq + the tokens we have seen so far...
        num_target_samples = len(target_ids.input_ids)

        if task_ids is not None:
            source_ids["task_ids"] = task_ids

        def generator_func():
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

        return DataStream(generator_func)


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
