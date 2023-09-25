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
from typing import List, Union

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
from .hf_auto_seq2seq_lm import HFAutoSeq2SeqLM

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
        use_seq2seq_tokenization: bool = False,
        chunk_size: int = 128,
        drop_remainder: bool = False,
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
            chunk_size: int
                unsigned int value to be used for chunk size.
            drop_remainder: bool
                Whether or not to keep the residual as an extra chunk if the
                total number of tokens is not divisible by the chunk size.

        Returns:
            DataStream[transformers.tokenization_utils_base.BatchEncoding]
                stream of encoded tokenization output corresponding to the input example.
        """
        if use_seq2seq_tokenization:
            return cls._forward_to_seq2seq_tokenization(
                example=example,
                tokenizer=tokenizer,
                max_source_length=max_source_length,
                max_target_length=max_target_length,
                verbalizer=verbalizer,
                task_ids=task_ids,
            )

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
        source_id_chunks = cls._force_to_batch_encoding_list_of_chunks(
            source_ids, target_ids, batched_mode, task_ids, chunk_size, drop_remainder
        )

        def generator_func():
            for chunk in source_id_chunks:
                yield chunk

        chunk_stream = DataStream(generator_func)
        # If it's batch mode, collapse down into one encoding batch object
        if batched_mode:
            return cls._collapse_stream_into_encoding(chunk_stream)
        # Otherwise just produce the stream to be chained
        return chunk_stream

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
    def _force_to_batch_encoding_list_of_chunks(
        source_ids: BatchEncoding,
        target_ids: BatchEncoding,
        batch_mode: bool,
        task_ids: Union[None, int],
        chunk_size: int,
        drop_remainder: bool,
    ) -> List[BatchEncoding]:
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
            List[BatchEncoding]
        """
        if not batch_mode:
            HFAutoCausalLM._concatenate_encodings(source_ids, target_ids)
            chunks = HFAutoCausalLM._split_encoding_into_chunks(
                encoding=source_ids,
                chunk_size=chunk_size,
                task_ids=task_ids,
            )
            return chunks
        # Otherwise we need to expand the dict along its keys,
        # mapping all of its encapsulated objects to new items.
        encodings = []
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
                new_encoding[key] = (
                    source_ids[key][batch_idx] + target_ids[key][batch_idx]
                )
            chunks = HFAutoCausalLM._split_encoding_into_chunks(
                encoding=new_encoding,
                chunk_size=chunk_size,
                drop_remainder=drop_remainder,
                task_ids=task_ids,
            )
            # Chunks are held as a list of lists
            encodings += chunks
        return encodings

    @staticmethod
    def _concatenate_encodings(left, right):
        for k in left.keys():
            left[k] = left[k] + right[k]

    @staticmethod
    def _split_encoding_into_chunks(
        encoding: dict, chunk_size: int, drop_remainder: bool = False, task_ids=None
    ):
        """Fetch the chunked batch encoding objects from source/target encoding(s).
        If no target encoding is provided, it's assumed that the source and target
        have already been concatenated.

        If drop remainder is enabled, do not yield uneven chunks. For now, this parameter
        is not exposed.
        """
        chunked_encodings = []
        # all encoding keys have the same length list values; we just use input ids
        tok_len = len(encoding["input_ids"])
        # Build a batch encoding for every chunk; for each data,
        # use the slice for all keys inside of the source_encoding.
        if tok_len >= chunk_size:
            slice_len = (tok_len // chunk_size) * chunk_size
            # If we have a remainder and we don't want to drop it, add a new chunk
            if not drop_remainder and slice_len != tok_len:
                slice_len += chunk_size
        # We just have one big chunk
        else:
            slice_len = tok_len
        chunked_encodings = [
            BatchEncoding(
                data={
                    k: v[chunk_num : chunk_num + chunk_size]
                    for k, v in encoding.items()
                }
            )
            for chunk_num in range(0, slice_len, chunk_size)
        ]
        for enc in chunked_encodings:
            enc["task_ids"] = task_ids
        return chunked_encodings

    @staticmethod
    def _collapse_stream_into_encoding(
        stream: DataStream[BatchEncoding],
    ) -> BatchEncoding:
        """Given a stream batch encodings, collapse them back into
        one encoding, i.e., the return value of the batch encoding.

        Args:
            streams: List[DataStream[BatchEncoding]]
                Objects to be collapsed into a single encoding.
            encoding_keys: dict_keys
                Dictionary keys to copy over from batch encoding list.

        Returns:
            BatchEncoding
                Collapsed batch encoding to be returned from tokenization func.
        """
        encoding_keys = None
        new_encoding = BatchEncoding()
        # Now build the individual lists lists for each entry
        for enc in stream:
            # Initialize the existing keys in the new encoding
            if encoding_keys is None:
                encoding_keys = enc.keys()
                for k in encoding_keys:
                    new_encoding[k] = []
            for k in encoding_keys:
                new_encoding[k].append(enc[k])
        return new_encoding

    @staticmethod
    def _forward_to_seq2seq_tokenization(
        example,
        tokenizer,
        max_source_length,
        max_target_length,
        verbalizer,
        task_ids,
    ):
        return HFAutoSeq2SeqLM.tokenize_function(
            example=example,
            tokenizer=tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            verbalizer=verbalizer,
            task_ids=task_ids,
        )
