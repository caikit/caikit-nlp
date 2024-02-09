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
import torch

# First Party
from caikit.core.data_model import DataStream
from caikit.core.exceptions import error_handler
from caikit.core.modules import module
import alog

# Local
from ...data_model import GenerationTrainRecord, PromptOutputModelType

# Note: Below module is imported to allow loading of fm stack sphinx models
from ...toolkit.text_generation import (  # pylint: disable=unused-import
    granite_modeling_llama,
)
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
        use_seq2seq_approach: bool = True,
        chunk_size: int = 128,
        drop_remainder: bool = False,
    ) -> Union[DataStream[BatchEncoding], BatchEncoding]:
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
            use_seq2seq_approach: bool
                Indicates whether or not we should use a sequence style approach
                or use chunking parameters.
            chunk_size: int
                unsigned int value to be used for chunk size.
                Only used if use_seq2seq_approach=True.
            drop_remainder: bool
                Whether or not to keep the residual as an extra chunk if the
                total number of tokens is not divisible by the chunk size.
                Only used if use_seq2seq_approach=True.

        Returns:
            Union[DataStream[BatchEncoding], BatchEncoding]
                stream of encoded tokenization output corresponding to the input example
                or a single batch encoding object containing 1+ tokenized results.
        """
        ### Things common to all Causal LM tokenization approaches
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
        # Treat this as a seq2seq type problem. Note that this implementation is different
        # from the seq2seq tokenization function even though it is conceptually similar due
        # to sequence length / padding requirements assumed internally by causal LMs.
        if use_seq2seq_approach:
            return cls._causal_lm_padding_as_seq2seq(
                tokenizer=tokenizer,
                source=source,
                target=target,
                max_source_length=max_source_length,
                max_target_length=max_target_length,
                task_ids=task_ids,
            )
        # Do causal language model chunking
        return cls._causal_lm_as_chunked(
            tokenizer=tokenizer,
            source=source,
            target=target,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            batched_mode=batched_mode,
            task_ids=task_ids,
            chunk_size=chunk_size,
            drop_remainder=drop_remainder,
        )

    def _get_data_collator(self, **kwargs) -> "transformers.DataCollator":
        """Function to return appropriate data collator based on resource.

        DataCollatorForLanguageModeling is used here which will dynamically
        padded to maximum length of a batch if they are not all of the same
        length.

        NOTE: If mlm (masked language modeling) is not passed in kwargs,
        this function will automatically set it to `False`.

        FIXME: This should be consolidated with what is in the prompt tuning
        module, which currently does its own collator management outside of the
        resource classes.

        Args:
            **kwargs:
                All the keyword arguments passed to this function
                will get filtered out to appropriate ones that are
                applicable to implemented data collator.
        Returns:
            transformers.DataCollator
                Collator to be used for causal language modeling.
        """

        applicable_args = ["mlm", "pad_to_multiple_of"]
        collator_kwargs = {key: kwargs[key] for key in applicable_args if key in kwargs}

        if "mlm" not in collator_kwargs:
            collator_kwargs["mlm"] = False

        return DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer, return_tensors="pt", **collator_kwargs
        )

    ### Tokenization strategy implementations
    # Chunked causal language modeling
    @classmethod
    def _causal_lm_as_chunked(
        cls,
        tokenizer: "AutoTokenizer",
        source: str,
        target: str,
        max_source_length: int,
        max_target_length: int,
        batched_mode: bool,
        task_ids: Union[None, int],
        chunk_size: int,
        drop_remainder: bool,
    ) -> Union[DataStream[BatchEncoding], BatchEncoding]:
        """Given a source and target string, build the chunked concatenated sequence and formulate
        the batch encoded chunks for the sequence. If running in batch mode, the chunks will be
        collapsed into a single batch encoding for the whole sequence. Otherwise, each chunk will
        placed in its own BatchEncoding and encapsulated within a datastream.

        Args:
            tokenizer: AutoTokenizer
                Tokenizer object to be applied to input records.
            source: str
                Raw source string.
            target: str
                Raw target string.
            max_source_length: int
                Maximum length for input sequences.
            max_target_length: int
                Maximum length for output sequences.
            batched_mode: bool
                Whether or not we should produce a stream of encodings or a single
                encoding representing all of the chunked sequence.
            task_ids: Union[None, int]
                Task IDs to be used for multiprompt tuning.
            chunk_size: int
                unsigned int value to be used for chunk size.
            drop_remainder: bool
                Whether or not to keep the residual as an extra chunk if the
                total number of tokens is not divisible by the chunk size.

        Returns:
            Union[DataStream[BatchEncoding], BatchEncoding]
                Encoded chunked sequence as a stream or batch encoding object.
        """
        source_ids = tokenizer(source, max_length=max_source_length, truncation=True)
        target_ids = tokenizer(target, max_length=max_target_length, truncation=True)

        # Force everything to a list of batch encodings; for non-batch mode, this just
        # puts it into a list. For batch mode, we get a list of batch encodings,
        # allowing us to standardize subsequent processing a bit.
        #
        # For example, given chunk size 2, we might have something like:
        # [
        #   {'input_ids': [31, 48], 'attention_mask': [1, 1]},
        #   {'input_ids': [47, 1], 'attention_mask': [1, 1]},
        #   ...
        # ]
        # (where the above objects are batch encodings, which are a subclass of dict)
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
        # NOTE: it might be a good idea to deprecate this to force standardization
        # onto using batch encodings the way that they are intended to be
        return chunk_stream

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
            chunk_size: int
                unsigned int value to be used for chunk size.
            drop_remainder: bool
                Whether or not to keep the residual as an extra chunk if the
                total number of tokens is not divisible by the chunk size.

        Returns:
            List[BatchEncoding]
                List of batch encodings, each of which encapsulates the contents
                of a single chunk.
        """
        if not batch_mode:
            HFAutoCausalLM._concatenate_encodings(source_ids, target_ids)
            chunks = HFAutoCausalLM._split_encoding_into_chunks(
                encoding=source_ids,
                chunk_size=chunk_size,
                drop_remainder=drop_remainder,
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
    def _concatenate_encodings(left: BatchEncoding, right: BatchEncoding) -> None:
        """Given two batch encodings, combine their entries into a single encoding.

        Args:
            left: BatchEncoding
                Encoding representing left sequence, which will be updated in place.
                Corresponds to source.
            right: BatchEncoding
                Encoding representing right sequence, which will be stacked onto the left
                encoding. Corresponds to target.
        """
        for k in left.keys():
            left[k].extend(right[k])

    @staticmethod
    def _split_encoding_into_chunks(
        encoding: BatchEncoding,
        chunk_size: int,
        drop_remainder: bool,
        task_ids: Union[None, int],
    ) -> List[BatchEncoding]:
        """Fetch the chunked batch encoding objects from the concatenated encoding.

        Args:
            encoding: BatchEncoding
                BatchEncoding holding the concatenated source/target for one example.
            chunk_size: int
                unsigned int value to be used for chunk size.
            drop_remainder: bool
                Whether or not to keep the residual as an extra chunk if the
                total number of tokens is not divisible by the chunk size.
            task_ids: Union[None, int]
                Optional task IDs for MPT to be propagated to produced encodings.

        Returns:
            List[BatchEncoding]
                List of encodings, where each encoding represents one chunk.
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

    # Causal language modeling as a sequence to sequence problem
    @staticmethod
    def _causal_lm_padding_as_seq2seq(
        tokenizer: "AutoTokenizer",
        source: str,
        target: str,
        max_source_length: int,
        max_target_length: int,
        task_ids: Union[None, int],
    ) -> BatchEncoding:
        """Tokenize the example as a seq2seq type problem; this is conceptually similar to
        what seq2seq tokenization is doing, but some care needs be taken to ensure the labels
        are the same length as the input sequence because of the shifting mechanism implemented
        in most causal language models.

        Collator compatability is extremely important here; because we are setting the labels
        directly, we should NOT use the causal lm collator, otherwise it will clobber it with a
        shifted input sequence.

        Args:
            tokenizer: AutoTokenizer
                Tokenizer object to be applied to input records.
            source: str
                Raw source string.
            target: str
                Raw target string.
            max_source_length: int
                Maximum length for input sequences.
            max_target_length: int
                Maximum length for output sequences.
            task_ids: Union[None, int]
                Optional task IDs for MPT to be propagated to produced encodings.
        Returns:
            BatchEncoding
                BatchEncoding object corresponding to this example, where the input_ids,
                attention_mask, and labels all have the same length, i.e.,
                [max_source_length + max_target_length + 1].
        """
        IGNORE_ID = -100
        # ID of the token to append after our target string; this should generally be pad / EOS
        FINAL_TOK_ID = tokenizer.eos_token_id
        max_concat_length = max_source_length + max_target_length + 1

        # Truncate based on max source or max target length before considering as a joined sequence
        model_inputs = tokenizer(source, truncation=True, max_length=max_source_length)
        labels = tokenizer(target, truncation=True, max_length=max_target_length + 1)

        # Combine the source + target strings into the source input IDs
        # This makes the source and target the same length, and then masks the source out of the
        # target IDs, and updates the length of the attention vector to be evenly spread on the
        # whole combined sequence
        sample_input_ids = model_inputs["input_ids"]
        label_input_ids = labels["input_ids"] + [FINAL_TOK_ID]
        model_inputs["input_ids"] = sample_input_ids + label_input_ids
        labels["input_ids"] = [IGNORE_ID] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])
        # Now we have to update everything to be the max length of the tokenizer, then pad &
        # ensure all of the padded stuff we have added has attention weights of 0.
        sample_input_ids = model_inputs[
            "input_ids"
        ]  # NOTE - combined source + target + <FINAL_TOK_ID>

        label_input_ids = labels["input_ids"]
        model_inputs = tokenizer.pad(
            model_inputs, padding="max_length", max_length=max_concat_length
        )

        if tokenizer.padding_side.lower() == "left":
            labels["input_ids"] = [IGNORE_ID] * (
                max_concat_length - len(sample_input_ids)
            ) + label_input_ids
        else:
            labels["input_ids"] = label_input_ids + [IGNORE_ID] * (
                max_concat_length - len(sample_input_ids)
            )

        model_inputs["input_ids"] = torch.tensor(
            model_inputs["input_ids"][:max_concat_length]
        )
        model_inputs["attention_mask"] = torch.tensor(
            model_inputs["attention_mask"][:max_concat_length]
        )

        labels["input_ids"] = torch.tensor(labels["input_ids"][:max_concat_length])
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["task_ids"] = task_ids
        return model_inputs
