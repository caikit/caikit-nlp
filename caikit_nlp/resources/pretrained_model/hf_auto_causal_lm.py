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
from copy import copy
from collections.abc import Mapping
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
            tokenizer: AutoTokenizer
                Tokenizer object to be applied to input records.
            max_source_length: int
                Maximum length for input sequences.
            max_target_length: int
                Maximum length for output sequences.
            verbalizer: Union[None, str]
                Verbalizer to be rendered into each text.
            task_ids: Union[None, int]
                Task IDs to be used for multiprompt tuning.

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
        
        source_ids = tokenizer(source, max_length=max_source_length, truncation=True)
        target_ids = tokenizer(target, max_length=max_target_length, truncation=True)
        # Force everything to a list of batch encodings; for non-batch mode, this just
        # puts it into a list. For batch mode, we get a list of batch encodings,
        # allowing us to standardize subsequent processing a bit.

        source_ids = cls._force_to_batch_encoding_list(
            source_ids,
            target_ids,
            batched_mode,
            task_ids
        )

        def build_generator_func(source_ids):
            def single_generator_func():
                for idx in range(source_ids["num_target_samples"]):
                    s = copy(source_ids)
                    s["attention_mask"] = cls._get_attention_mask(
                        source_ids, 
                        idx
                    )
                    yield s
            return single_generator_func

        if not batched_mode:
            return DataStream(build_generator_func(source_ids))
        streams = [DataStream(build_generator_func(s)) for s in source_ids]
        encoding_keys = source_ids[0].keys()
        return cls._collapse_streams_into_encoding(streams, encoding_keys)

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

    def _force_to_batch_encoding_list(source_ids, target_ids, batch_mode, task_ids):
        if not batch_mode:
            source_ids["input_ids"] = source_ids.input_ids + target_ids.input_ids
            source_ids["num_target_samples"] = len(target_ids.input_ids)
            source_ids["task_ids"] = task_ids            
            return source_ids
        # Otherwise we need to expand the dict along its keys, 
        # mapping all of its encapsulated objects to new items.
        encodings = []
        id_keys = source_ids.keys()
        for batch_idx in range(len(source_ids.input_ids)):
            new_encoding = BatchEncoding()
            for key in id_keys:
                if key == "input_ids":
                    new_encoding[key] = source_ids[key][batch_idx] + target_ids[key][batch_idx]
                else:
                    new_encoding[key] = source_ids[key][batch_idx]
            new_encoding["num_target_samples"] = len(target_ids[key][batch_idx])
            new_encoding["task_ids"] = task_ids
            encodings.append(new_encoding)
        return encodings

    def _get_attention_mask(source_ids, idx):
        return (
            source_ids["attention_mask"]
            + [1] * (idx + 1)
            + [0] * (source_ids["num_target_samples"] - idx - 1)
        )

    @classmethod
    def _collapse_streams_into_encoding(cls, streams, encoding_keys):
        new_encoding = BatchEncoding()
        for k in encoding_keys:
            new_encoding[k] = []
        # Now build the individual lists lists for each entry
        for stream in streams:
            for enc in stream:
                for k in encoding_keys:
                    new_encoding[k].append(enc[k])
        return new_encoding
