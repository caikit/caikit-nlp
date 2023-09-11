"""Tests for all pretrained model resource types; since most of the functionality for
these classes live through the base class, we test them all in this file.

Importantly, these tests do NOT depend on access to external internet to run correctly.

NOTE: If the subclasses become sufficiently complex, this test file should be split up.
"""
# Standard
from unittest.mock import patch

# Third Party
from datasets import IterableDataset as TransformersIterableDataset
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


### Tests for tokenization behaviors
SAMPLE_TRAINING_DATA = caikit.core.data_model.DataStream.from_iterable(
    [
        GenerationTrainRecord(input="This is a sequence", output="This is a target"),
        GenerationTrainRecord(input="I am just a little", output="bit tired"),
    ]
)


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


def test_causal_lm_tok_output_correctness(models_cache_dir):
    """Validate the correctness of the attention mask for the language modeling objective."""
    causal_lm = HFAutoCausalLM.bootstrap(
        model_name=CAUSAL_LM_MODEL, tokenizer_name=CAUSAL_LM_MODEL
    )
    sample = GenerationTrainRecord(
        input="This len does not matter", output="but this one does!"
    )
    (tok_func, _) = causal_lm.build_task_tokenize_closure(
        tokenizer=causal_lm.tokenizer,
        max_source_length=100,
        max_target_length=100,
        verbalizer="{{input}}",
        task_ids=0,
    )
    input_tok = causal_lm.tokenizer.encode(sample.input)
    output_tok = causal_lm.tokenizer.encode(sample.output)
    tok_stream = tok_func(sample)
    # Ensure we get one token per output in our stream
    assert isinstance(tok_stream, caikit.core.data_model.DataStream)
    assert len(tok_stream) == len(output_tok)
    for idx, tok_sample in enumerate(tok_stream):
        # We expect by default, everything is in order, and each attention mask grows the tokens
        # we attend to in the target by one, until we are paying attention to the whole sequence.
        expected_target_mask = torch.tensor(
            ([1] * (idx + 1)) + [0] * (len(output_tok) - idx - 1)
        )
        actual_target_mask = torch.tensor(
            tok_sample["attention_mask"][-len(output_tok) :]
        )
        assert bool(torch.all(expected_target_mask == actual_target_mask))
        # Check the source mask; we should always attend to the whole source sequence
        actual_source_mask = torch.tensor(
            tok_sample["attention_mask"][: len(input_tok)]
        )
        assert bool(torch.all(torch.tensor([1] * len(input_tok)) == actual_source_mask))
        # Also, the number of tokens we attend to should be the sum of toks in input/output
        assert (len(actual_target_mask) + len(actual_source_mask)) == len(
            tok_sample["attention_mask"]
        )
        # Ensure we support MPT
        assert hasattr(tok_sample, "task_ids")
        assert tok_sample["task_ids"] == 0


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
