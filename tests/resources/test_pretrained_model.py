"""Tests for all pretrained model resource types; since most of the functionality for
these classes live through the base class, we test them all in this file.

Importantly, these tests do NOT depend on access to external internet to run correctly.

NOTE: If the subclasses become sufficiently complex, this test file should be split up.
"""
# Standard
from unittest.mock import patch

# Third Party
from datasets import IterableDataset as TransformersIterableDataset
from torch.utils.data import DataLoader
import pytest
import torch
import transformers

# First Party
from caikit.core.data_model import DataStream
import aconfig
import caikit

# Local
from caikit_nlp.data_model import GenerationTrainRecord
from caikit_nlp.resources.pretrained_model import HFAutoCausalLM, HFAutoSeq2SeqLM
from tests.fixtures import (
    CAUSAL_LM_MODEL,
    SEQ2SEQ_LM_MODEL,
    models_cache_dir,
    temp_cache_dir,
)


def test_boostrap_causal_lm(models_cache_dir):
    """Ensure that we can bootstrap a causal LM if we have download access."""
    # If we have an empty cachedir & do allow downloads, we should be able to init happily
    base_model = HFAutoCausalLM.bootstrap(
        model_name=CAUSAL_LM_MODEL, tokenizer_name=CAUSAL_LM_MODEL
    )
    assert isinstance(base_model, HFAutoCausalLM)
    assert base_model.MODEL_TYPE is transformers.AutoModelForCausalLM
    assert base_model.TASK_TYPE == "CAUSAL_LM"


def test_boostrap_causal_lm_override_dtype(models_cache_dir):
    """Ensure that we can override the data type to init the model with at bootstrap time."""
    # If we have an empty cachedir & do allow downloads, we should be able to init happily,
    # and if we provide a data type override, that override should be applied at bootstrap time.
    base_model = HFAutoCausalLM.bootstrap(
        model_name=CAUSAL_LM_MODEL,
        tokenizer_name=CAUSAL_LM_MODEL,
        torch_dtype=torch.float64,
    )
    assert isinstance(base_model, HFAutoCausalLM)
    assert base_model.MODEL_TYPE is transformers.AutoModelForCausalLM
    assert base_model.TASK_TYPE == "CAUSAL_LM"
    assert base_model.model.dtype is torch.float64


@patch("transformers.models.auto.tokenization_auto.AutoTokenizer.from_pretrained")
def test_boostrap_causal_lm_download_disabled(mock_tok_from_pretrained, temp_cache_dir):
    """Ensure that we can't try to download if downloads are disabled"""
    # NOTE: allow_downloads is false by default
    mock_tok_from_pretrained.side_effect = RuntimeError("It's a mock")
    with pytest.raises(RuntimeError):
        HFAutoCausalLM.bootstrap(model_name="foo/bar/baz", tokenizer_name="foo/bar/baz")
    kwargs = mock_tok_from_pretrained.call_args[1]
    assert kwargs["local_files_only"] == True


@patch("transformers.models.auto.tokenization_auto.AutoTokenizer.from_pretrained")
def test_boostrap_causal_lm_download_enabled(mock_tok_from_pretrained, temp_cache_dir):
    """Ensure that we can try to download if downloads are enabled."""
    with patch(
        "caikit_nlp.resources.pretrained_model.base.get_config",
        return_value=aconfig.Config({"allow_downloads": True}),
    ):
        mock_tok_from_pretrained.side_effect = RuntimeError("It's a mock")
        with pytest.raises(RuntimeError):
            HFAutoCausalLM.bootstrap(
                model_name="foo/bar/baz", tokenizer_name="foo/bar/baz"
            )
        kwargs = mock_tok_from_pretrained.call_args[1]
        assert kwargs["local_files_only"] == False


def test_boostrap_model_path(models_cache_dir):
    """Ensure that we can bootstrap works with loading a model from local directory"""
    # If we have an empty cachedir & do allow downloads, we should be able to init happily
    base_model = HFAutoCausalLM.bootstrap(
        model_name=CAUSAL_LM_MODEL,
    )
    assert isinstance(base_model, HFAutoCausalLM)
    assert base_model.MODEL_TYPE is transformers.AutoModelForCausalLM
    assert base_model.TASK_TYPE == "CAUSAL_LM"


### Tests for tokenization behaviors
SAMPLE_TRAINING_DATA = caikit.core.data_model.DataStream.from_iterable(
    [
        GenerationTrainRecord(input="This is a sequence", output="This is a target"),
        GenerationTrainRecord(input="I am just a little", output="bit tired"),
    ]
)

# Causal LM tokenization strategies
### 1. Tests for Causal LM tokenization chunking
def test_causal_lm_tokenize_func_contains_wrapped_stream(models_cache_dir):
    """Ensure the Causal LM tokenize func produces a wrapped stream that can be flattened."""
    causal_lm = HFAutoCausalLM.bootstrap(
        model_name=CAUSAL_LM_MODEL, tokenizer_name=CAUSAL_LM_MODEL
    )
    (tok_func, requires_unwrapping) = causal_lm.build_task_tokenize_closure(
        tokenizer=causal_lm.tokenizer,
        max_source_length=100,
        max_target_length=100,
        verbalizer="{{input}}",
        use_seq2seq_approach=False,
    )
    map_stream = SAMPLE_TRAINING_DATA.map(tok_func)
    # Since tok_func for causal lm creates a datastream, we should get a stream
    # back; make sure that it's a stream of streams before proceeding
    assert requires_unwrapping is True
    assert isinstance(map_stream, caikit.core.data_model.DataStream)
    assert isinstance(map_stream.peek(), caikit.core.data_model.DataStream)
    # Ensure that we can unwrap the stream
    unwrapped_stream = map_stream.flatten()
    assert isinstance(
        unwrapped_stream.peek(), transformers.tokenization_utils_base.BatchEncoding
    )


# Key cases here are:
# 1 - simplest and minimal case
# 3 - because the concat sequence is length 17, so we have a remainder
# 100 - which is much larger than the concatenated seq and should yield one chunk
@pytest.mark.parametrize(
    "chunk_size,drop_remainder",
    [(1, True), (1, False), (3, True), (3, False), (100, True), (100, False)],
)
def test_causal_lm_tok_output_correctness(models_cache_dir, chunk_size, drop_remainder):
    """Validate the tokenized results for the chunked language modeling objective."""
    causal_lm = HFAutoCausalLM.bootstrap(
        model_name=CAUSAL_LM_MODEL, tokenizer_name=CAUSAL_LM_MODEL
    )
    sample = GenerationTrainRecord(
        input="Hello world", output="How are you doing today?!"
    )
    (tok_func, _) = causal_lm.build_task_tokenize_closure(
        tokenizer=causal_lm.tokenizer,
        max_source_length=100,
        max_target_length=100,
        verbalizer="{{input}}",
        task_ids=0,
        use_seq2seq_approach=False,
        chunk_size=chunk_size,
        drop_remainder=drop_remainder,
    )
    input_tok = causal_lm.tokenizer.encode(sample.input)
    output_tok = causal_lm.tokenizer.encode(sample.output)
    concat_tok = input_tok + output_tok
    tok_stream = tok_func(sample)
    # Ensure we get one token per output in our stream
    assert isinstance(tok_stream, caikit.core.data_model.DataStream)
    # Figure out how many chunks we should have, including if we have a remainder
    has_remainder = False
    if len(concat_tok) > chunk_size:
        num_expected_chunks = len(concat_tok) // chunk_size
        # Should only care about the remainder if we are not dropping it
        if num_expected_chunks * chunk_size != len(concat_tok) and not drop_remainder:
            has_remainder = True
    else:
        num_expected_chunks = 1
        chunk_size = len(concat_tok)
    tok_list = list(tok_stream)
    assert len(tok_list) == num_expected_chunks + has_remainder
    # Check all full chunks. Note that we always attend to everything
    for idx in range(num_expected_chunks):
        assert len(tok_list[idx]["attention_mask"]) == chunk_size
        assert len(tok_list[idx]["input_ids"]) == chunk_size
        assert all(atn == 1 for atn in tok_list[idx]["attention_mask"])
        assert tok_list[idx]["task_ids"] == 0
    # Check the remainder; lists should be the same length, but less than the chunk size
    if has_remainder:
        remainder = tok_list[-1]
        assert len(remainder["attention_mask"]) == len(remainder["input_ids"])
        assert len(remainder["input_ids"]) < chunk_size
        assert all(atn == 1 for atn in remainder["attention_mask"])


def test_causal_lm_batch_tokenization(models_cache_dir):
    """Ensure that we can batch process causal lm inputs correctly."""
    causal_lm = HFAutoCausalLM.bootstrap(
        model_name=CAUSAL_LM_MODEL, tokenizer_name=CAUSAL_LM_MODEL
    )
    train_stream = DataStream.from_iterable(
        [
            GenerationTrainRecord(input="hello there", output="world"),
            GenerationTrainRecord(input="how", output="today"),
        ]
    )
    fn_kwargs = {
        "tokenizer": causal_lm.tokenizer,
        "max_source_length": 10,
        "max_target_length": 10,
        "use_seq2seq_approach": False,
    }
    # Create an iterable dataset by batching...
    def get(train_stream):
        for data in train_stream:
            yield {"input": data.input, "output": data.output}

    dataset = TransformersIterableDataset.from_generator(
        get, gen_kwargs={"train_stream": train_stream}
    )
    batched_dataset = dataset.map(
        causal_lm.tokenize_function,
        fn_kwargs=fn_kwargs,
        batched=True,
        remove_columns=["input", "output"],
    )

    # Do the same thing with no batching via tokenize closure + unwrapping
    tok_func = causal_lm.build_task_tokenize_closure(**fn_kwargs)[0]
    mapped_indiv_stream = train_stream.map(tok_func).flatten()
    for indiv_res, batched_res in zip(mapped_indiv_stream, batched_dataset):
        # All keys should match (input ids, attention mask)
        assert indiv_res.keys() == batched_res.keys()
        # And all of their values should be the same
        for k in indiv_res:
            assert indiv_res[k] == batched_res[k]


### 2. Tests for causal LM framed as a seq2seq problem
# NOTE: For these tests, we should be careful to always test left and right padding
@pytest.mark.parametrize(
    "padding_side",
    ["left", "right"],
)
def test_causal_lm_as_a_sequence_problem_no_truncation(models_cache_dir, padding_side):
    causal_lm = HFAutoCausalLM.bootstrap(
        model_name=CAUSAL_LM_MODEL, tokenizer_name=CAUSAL_LM_MODEL
    )
    sample = GenerationTrainRecord(
        input="Hello world", output="How are you doing today?!"
    )
    max_lengths = 20
    # First, build the output we expect for left / right respectively...
    input_tok = causal_lm.tokenizer.encode(sample.input)
    output_tok = causal_lm.tokenizer.encode(sample.output) + [
        causal_lm.tokenizer.eos_token_id
    ]
    concat_res = input_tok + output_tok
    masked_res = ([-100] * len(input_tok)) + output_tok

    # This must true because otherwise no padding was needed, e.g., truncation
    assert len(input_tok) < max_lengths
    assert len(output_tok) < (max_lengths + 1)
    pads_needed = (1 + 2 * max_lengths) - len(concat_res)
    if causal_lm.tokenizer.padding_side.lower() == "left":
        expected_input_ids = torch.tensor(
            [causal_lm.tokenizer.pad_token_id] * pads_needed + concat_res
        )
        expected_attn_mask = torch.tensor([0] * pads_needed + [1] * len(concat_res))
        expected_labels = torch.tensor([-100] * pads_needed + masked_res)
    else:
        expected_input_ids = torch.tensor(
            concat_res + [causal_lm.tokenizer.pad_token_id] * pads_needed
        )
        expected_attn_mask = torch.tensor([1] * len(concat_res) + [0] * pads_needed)
        expected_labels = torch.tensor(masked_res + [-100] * pads_needed)

    # Now build the analogous tokenizer closure and compare the tensors
    (tok_func, _) = causal_lm.build_task_tokenize_closure(
        tokenizer=causal_lm.tokenizer,
        max_source_length=max_lengths,
        max_target_length=max_lengths,
        verbalizer="{{input}}",
        task_ids=0,
        use_seq2seq_approach=True,
    )
    tok_res = tok_func(sample)
    assert tok_res["task_ids"] == 0
    assert torch.all(tok_res["input_ids"] == expected_input_ids)
    assert torch.all(tok_res["attention_mask"] == expected_attn_mask)
    assert torch.all(tok_res["labels"] == expected_labels)


### Tests for Seq2Seq tokenization
def test_seq2seq_tokenize_func_contains_unwrapped_stream(models_cache_dir):
    """Ensure the seq2seq tokenizer produces an unwrapped stream; not flattening needed."""
    seq2seq = HFAutoSeq2SeqLM.bootstrap(
        model_name=SEQ2SEQ_LM_MODEL, tokenizer_name=SEQ2SEQ_LM_MODEL
    )
    (tok_func, requires_unwrapping) = seq2seq.build_task_tokenize_closure(
        tokenizer=seq2seq.tokenizer,
        max_source_length=100,
        max_target_length=100,
        verbalizer="{{input}}",
        task_ids=0,
    )
    map_stream = SAMPLE_TRAINING_DATA.map(tok_func)
    # Since we don't require unwrapping, i.e., each input sample just produces 1,
    # result, we should just get a stream of batch encodings we can use directly.
    assert requires_unwrapping is False
    assert isinstance(map_stream, caikit.core.data_model.DataStream)
    assert isinstance(
        map_stream.peek(), transformers.tokenization_utils_base.BatchEncoding
    )


def test_seq2seq_tok_output_correctness(models_cache_dir):
    """Validate the correctness of the attention mask for the seq2seq task."""
    seq2seq = HFAutoSeq2SeqLM.bootstrap(
        model_name=SEQ2SEQ_LM_MODEL, tokenizer_name=SEQ2SEQ_LM_MODEL
    )
    sample = GenerationTrainRecord(
        input="This len does not matter", output="and this one doesn't either!"
    )
    (tok_func, _) = seq2seq.build_task_tokenize_closure(
        tokenizer=seq2seq.tokenizer,
        max_source_length=20,
        max_target_length=20,
        verbalizer="{{input}}",
        task_ids=0,
    )
    input_tok = seq2seq.tokenizer.encode(sample.input)

    tok_sample = tok_func(sample)
    # Ensure we get one seq2seq; i.e., the result should NOT be a stream,
    # and we should only be attending to the tokens from the input sequence.
    assert isinstance(tok_sample, transformers.tokenization_utils_base.BatchEncoding)
    assert sum(tok_sample["attention_mask"]) == len(input_tok)
    # Ensure we support MPT
    assert hasattr(tok_sample, "task_ids")
    assert tok_sample["task_ids"] == 0


### Tests for collator compatability
# These tests should validate that we can use our tokenization function to
# build torch loaders around datasets using different collators.
# TODO: Expand to cover transformer datasets, i.e., what is produced by
# text gen preprocessing functions. For now, they only check the minimal
# case with the default data collator.
@pytest.mark.parametrize(
    "collator_fn",
    [transformers.default_data_collator],
)
def test_loader_can_batch_list_of_seq2seq_outputs(collator_fn):
    # Build the dataset
    train_stream = DataStream.from_iterable(
        [
            GenerationTrainRecord(input="hello world", output="how are you today?"),
            GenerationTrainRecord(input="goodbye", output="world"),
            GenerationTrainRecord(input="good morning", output="have a good day"),
            GenerationTrainRecord(input="good night", output="have nice dreams"),
        ]
    )
    seq2seq = HFAutoSeq2SeqLM.bootstrap(
        model_name=SEQ2SEQ_LM_MODEL, tokenizer_name=SEQ2SEQ_LM_MODEL
    )
    (tok_func, _) = seq2seq.build_task_tokenize_closure(
        tokenizer=seq2seq.tokenizer,
        max_source_length=20,
        max_target_length=20,
        verbalizer="{{input}}",
        task_ids=0,
    )
    tok_results = [tok_func(x) for x in list(train_stream)]
    dl = DataLoader(
        tok_results,
        shuffle=False,
        batch_size=2,
        collate_fn=collator_fn,
    )
    # Loader should create 2 batches
    loader_list = list(dl)
    assert len(loader_list) == 2


@pytest.mark.parametrize(
    "collator_fn",
    [transformers.default_data_collator],
)
def test_loader_can_batch_list_of_causal_lm_outputs(collator_fn):
    # Build the dataset
    train_stream = DataStream.from_iterable(
        [
            GenerationTrainRecord(input="hello world", output="how are you today?"),
            GenerationTrainRecord(input="goodbye", output="world"),
            GenerationTrainRecord(input="good morning", output="have a good day"),
            GenerationTrainRecord(input="good night", output="have nice dreams"),
        ]
    )
    causal_lm = HFAutoCausalLM.bootstrap(
        model_name=CAUSAL_LM_MODEL, tokenizer_name=CAUSAL_LM_MODEL
    )
    (tok_func, _) = causal_lm.build_task_tokenize_closure(
        tokenizer=causal_lm.tokenizer,
        max_source_length=20,
        max_target_length=20,
        verbalizer="{{input}}",
        task_ids=0,
        use_seq2seq_approach=True,
    )
    tok_results = [tok_func(x) for x in list(train_stream)]
    dl = DataLoader(
        tok_results,
        shuffle=False,
        batch_size=2,
        collate_fn=collator_fn,
    )
    # Loader should create 2 batches
    loader_list = list(dl)
    assert len(loader_list) == 2
