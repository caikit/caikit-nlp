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
from copy import copy
from typing import Callable, List, Tuple, Union

# Third Party
from transformers import (
    AutoModelForCausalLM,
    BatchEncoding,
    DataCollatorForLanguageModeling,
)
from transformers.models.auto import modeling_auto

# First Party
from caikit.core.data_model import DataStream
from caikit.core.exceptions import error_handler
from caikit.core.modules import module
import alog

# Local
from ...data_model import GenerationTrainRecord, PromptOutputModelType
from ...toolkit.verbalizer_utils import render_verbalizer
from .base import PretrainedModelBase

log = alog.use_channel("HFRCLM")
error = error_handler.get(log)


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
    ) -> DataStream[BatchEncoding]:
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
        source_ids, num_target_samples = cls._force_to_batch_encoding_list(
            source_ids, target_ids, batched_mode, task_ids
        )

        def build_generator_func(
            source_ids: BatchEncoding, num_target_samples: int
        ) -> Callable:
            """Builds a generator that can be applied to a single batch encoding and its
            corresponding original number of target samples.

            source_ids: BatchEncoding
                Source ID to generate different samples from.
            num_target_samples: int
                Number of target IDs; used for attention mask creation.
            """

            def single_generator_func():
                for idx in range(num_target_samples):
                    ret_source_ids = copy(source_ids)
                    ret_source_ids["attention_mask"] = cls._get_attention_mask(
                        source_ids,
                        idx,
                        num_target_samples,
                    )
                    yield ret_source_ids

            return single_generator_func

        if not batched_mode:
            return DataStream(build_generator_func(source_ids, num_target_samples))
        streams = [
            DataStream(build_generator_func(s_ids, n_target_samples))
            for s_ids, n_target_samples in zip(source_ids, num_target_samples)
        ]
        encoding_keys = source_ids[0].keys()
        return cls._collapse_streams_into_encoding(streams, encoding_keys)

    def _get_data_collator(self, **kwargs) -> "transformers.DataCollator":
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

    @staticmethod
    def _force_to_batch_encoding_list(
        source_ids: BatchEncoding,
        target_ids: BatchEncoding,
        batch_mode: bool,
        task_ids: Union[None, int],
    ) -> Tuple[Union[BatchEncoding, List[BatchEncoding]], Union[int, List[int]]]:
        """Forces our inputs into either a single batch encoding (if we aren't running in batch
        mode), or a list of Batch Encodings. I.e., a list of dicts instead of a dict of lists.
        The primary reason that we do this is to allow us to easily map a common generator
        func, regardless of whether or not we're running in batched mode.

        Args:
            source_ids: BatchEncoding
                Source ID batch encoding; target ID info will be merged into this one.
            target_ids: BatchEncoding
                Target ID batch encoding; information will be merged into source ID encoding.
            batch_mode: bool
                Whether or not we are processing a batch.
            task_ids: Union[None, int]
                Optional task IDs for MPT to be propagated to produced encodings.

        Returns:
            Tuple[Union[BatchEncoding, List[BatchEncoding]], Union[int, List]]
        """
        if not batch_mode:
            source_ids["input_ids"] = source_ids.input_ids + target_ids.input_ids
            source_ids["task_ids"] = task_ids
            num_target_samples = len(target_ids.input_ids)
            return source_ids, num_target_samples
        # Otherwise we need to expand the dict along its keys,
        # mapping all of its encapsulated objects to new items.
        encodings = []
        num_target_samples = []
        id_keys = source_ids.keys()
        key = None
        error.value_check(
            "<NLP94411004E>",
            source_ids.keys(),
            "Source ID batch encoding must have keys",
        )
        for batch_idx in range(len(source_ids.input_ids)):
            new_encoding = BatchEncoding()
            for key in id_keys:
                if key == "input_ids":
                    new_encoding[key] = (
                        source_ids[key][batch_idx] + target_ids[key][batch_idx]
                    )
                else:
                    new_encoding[key] = source_ids[key][batch_idx]
            num_target_samples.append(len(target_ids[key][batch_idx]))
            new_encoding["task_ids"] = task_ids
            encodings.append(new_encoding)
        return encodings, num_target_samples

    @staticmethod
    def _get_attention_mask(
        source_ids: BatchEncoding, idx: int, num_target_samples: int
    ) -> List[int]:
        """Get the attention mask for a given target token from some source encoding.

        Args:
            source_ids: BatchEncoding
                Source encoding that requires an attention mask.
            idx: int
                Index of the output token we attend up to.
            num_target_samples: int
                Length of the original target seequence being considered.

        Returns:
            List[int]
                Binary attention mask.
        """
        return (
            source_ids["attention_mask"]
            + [1] * (idx + 1)
            + [0] * (num_target_samples - idx - 1)
        )

    @staticmethod
    def _collapse_streams_into_encoding(
        streams: List[DataStream[BatchEncoding]], encoding_keys: "dict_keys"
    ) -> BatchEncoding:
        """Given a list of streams of batch encodings, collapse them back into
        one encoding, i.e., the return value of the batch encoding.

        Args:
            streams: List[DataStream[BatchEncoding]]
                Objects to be collapsed into a single encoding.
            encoding_keys: dict_keys
                Dictionary keys to copy over from batch encoding list.

        Returns:
            BatchEncoding
                Collapsed batch encoding to be returned from tokenizatino func.
        """
        new_encoding = BatchEncoding()
        for k in encoding_keys:
            new_encoding[k] = []
        # Now build the individual lists lists for each entry
        for stream in streams:
            for enc in stream:
                for k in encoding_keys:
                    new_encoding[k].append(enc[k])
        return new_encoding
