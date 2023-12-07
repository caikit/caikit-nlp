"""Tests for prompt tuning via PEFT.

NOTE: Currently these tests mix unit tests & regression tests.
We should be aware of the performance implications here, especially
if we start running the tests on CPUs in our CI; we'll likely want
to separate these in the future.
"""
# Standard
from typing import Iterable
from unittest import mock
import os
import tempfile

# Third Party
import pytest
import torch

# First Party
from caikit.interfaces.nlp.data_model import (
    ClassificationTrainRecord,
    GeneratedTextResult,
    GeneratedTextStreamResult,
)
import caikit

# Local
from caikit_nlp.data_model import ExponentialDecayLengthPenalty
from caikit_nlp.modules.text_generation import PeftPromptTuning
from caikit_nlp.modules.text_generation.peft_prompt_tuning import TuningType
from tests.fixtures import (
    causal_lm_dummy_model,
    causal_lm_train_kwargs,
    seq2seq_lm_dummy_model,
    seq2seq_lm_train_kwargs,
    set_cpu_device,
    temp_config,
)
import caikit_nlp

# Indexes into the peft config dictionary to get the actual prompt tuning config
DEFAULT_ADAPTER = "default"


### Tests validating block interfaces and behavior
def test_save_and_reload_with_base_model(causal_lm_dummy_model, set_cpu_device):
    """Ensure that we can save a model + its base to a tempdir and reload it."""
    with tempfile.TemporaryDirectory() as model_dir:
        causal_lm_dummy_model.save(model_dir, save_base_model=True)
        reloaded_config = caikit.core.ModuleConfig.load(os.path.abspath(model_dir))
        # We should have an exported decoder, but no encoder]
        assert os.path.isfile(os.path.join(model_dir, reloaded_config.DECODER))
        assert reloaded_config.ENCODER == ""
        reloaded_model = caikit_nlp.load(model_dir, torch_dtype="float16")
        assert reloaded_model.model.dtype is torch.float16
    assert isinstance(
        reloaded_model, caikit_nlp.modules.text_generation.PeftPromptTuning
    )


def test_save_and_reload_without_base_model(causal_lm_dummy_model):
    """Ensure that if we don't save the base model, we get the expected behavior."""
    with tempfile.TemporaryDirectory() as model_dir:
        causal_lm_dummy_model.save(model_dir, save_base_model=False)
        # For now, if we are missing the base model at load time, we throw ValueError
        with pytest.raises(ValueError):
            caikit_nlp.load(model_dir)


def test_save_log_loss_file(causal_lm_dummy_model):
    """Ensure saving a model saves the log loss file"""
    with tempfile.TemporaryDirectory() as model_dir:
        causal_lm_dummy_model.save(model_dir, save_base_model=False)
        file_path = os.path.join(
            model_dir,
            caikit_nlp.modules.text_generation.peft_prompt_tuning.TRAINING_LOSS_LOG_FILENAME,
        )

        assert os.path.isfile(file_path)


def test_run_model(causal_lm_dummy_model):
    """Ensure that we can run a model and get the right type out."""
    pred = causal_lm_dummy_model.run("This text doesn't matter")
    assert isinstance(pred, GeneratedTextResult)


def test_run_stream_out_model(causal_lm_dummy_model):
    """Ensure that we can run output streaming on a model and get the right type out."""
    pred_stream = causal_lm_dummy_model.run_stream_out("This text doesn't matter")
    assert isinstance(pred_stream, Iterable)
    for pred in pred_stream:
        print(pred)
        assert isinstance(pred, GeneratedTextStreamResult)


def test_verbalizer_rendering(causal_lm_dummy_model, monkeypatch):
    """Ensure that our model renders its verbalizer text correctly before calling tokenizer."""
    # Mock the tokenizer; we want to make sure its inputs are rendered properly
    monkeypatch.setattr(
        causal_lm_dummy_model,
        "tokenizer",
        mock.Mock(
            side_effect=RuntimeError("Tokenizer is a mock!"),
            # Set eos token property to be attribute of tokenizer
            eos_token="</s>",
        ),
    )
    input_text = "This text doesn't matter"
    causal_lm_dummy_model.verbalizer = " | {{input}} |"
    expected_tok_input = causal_lm_dummy_model.verbalizer.replace(
        "{{input}}", input_text
    )
    with pytest.raises(RuntimeError):
        causal_lm_dummy_model.run(input_text)
    # Our tokenizer mock should have run one time before our runtime error side effect
    assert len(causal_lm_dummy_model.tokenizer.call_args_list) == 1
    actual_tok_input = causal_lm_dummy_model.tokenizer.call_args_list[0].args[0]
    assert actual_tok_input == expected_tok_input


def test_verbalizer_cannot_be_static(causal_lm_train_kwargs):
    """Ensure that we throw an error if the verbalizer has no template (static model inputs)."""
    # NOTE: num_epochs 0 is intentional here; we should not need to rely on iterating over
    # our training data to determine that our verbalizer is invalid, we should do that before
    # we enter our training loop.
    patch_kwargs = {
        "verbalizer": "This has no template, so it always renders to the same thing..."
    }
    causal_lm_train_kwargs.update(patch_kwargs)
    with pytest.raises(ValueError):
        caikit_nlp.modules.text_generation.PeftPromptTuning.train(
            **causal_lm_train_kwargs
        )


def test_train_model(causal_lm_train_kwargs, set_cpu_device):
    """Ensure that we can train a model on some toy data for 1+ steps & run inference."""
    patch_kwargs = {
        "num_epochs": 1,
        "verbalizer": "Tweet text : {{input}} Label : ",
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                caikit_nlp.data_model.GenerationTrainRecord(
                    input="@foo what a cute dog!", output="no complaint"
                ),
                caikit_nlp.data_model.GenerationTrainRecord(
                    input="@bar this is the worst idea ever.", output="complaint"
                ),
            ]
        ),
        "torch_dtype": torch.bfloat16,
        "device": "cpu",
    }
    causal_lm_train_kwargs.update(patch_kwargs)
    model = caikit_nlp.modules.text_generation.PeftPromptTuning.train(
        **causal_lm_train_kwargs
    )
    # Test fallback to float32 behavior if this machine doesn't support bfloat16
    assert model.model.dtype is torch.float32
    # Ensure that we can get something out of it
    pred = model.run("@bar what a cute cat!")
    assert isinstance(pred, GeneratedTextResult)


def test_gen_trained_mpt(causal_lm_train_kwargs, set_cpu_device):
    """Ensure that we are able to do generation on causal-lm model trained
    using MPT."""
    patch_kwargs = {
        "num_epochs": 1,
        "verbalizer": "Tweet text : {{input}} Label : ",
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                caikit_nlp.data_model.GenerationTrainRecord(
                    input="@foo what a cute dog!", output="no complaint"
                ),
                caikit_nlp.data_model.GenerationTrainRecord(
                    input="@bar this is the worst idea ever.", output="complaint"
                ),
            ]
        ),
        "torch_dtype": torch.float32,
        "tuning_type": "MULTITASK_PROMPT_TUNING",
        "device": "cpu",
    }
    causal_lm_train_kwargs.update(patch_kwargs)
    model = caikit_nlp.modules.text_generation.PeftPromptTuning.train(
        **causal_lm_train_kwargs
    )
    # Ensure that we can get something out of it
    pred = model.run("@bar what a cute cat!")
    assert isinstance(pred, GeneratedTextResult)


def test_train_model_classification_record(causal_lm_train_kwargs, set_cpu_device):
    """Ensure that we can train a model on some toy data for 1+ steps & run inference."""
    patch_kwargs = {
        "num_epochs": 1,
        "verbalizer": "Tweet text : {{input}} Label : ",
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                ClassificationTrainRecord(
                    text="@foo what a cute dog!", labels=["no complaint"]
                ),
                ClassificationTrainRecord(
                    text="@bar this is the worst idea ever.", labels=["complaint"]
                ),
            ]
        ),
        "torch_dtype": torch.bfloat16,
        "device": "cpu",
    }
    causal_lm_train_kwargs.update(patch_kwargs)
    model = caikit_nlp.modules.text_generation.PeftPromptTuning.train(
        **causal_lm_train_kwargs
    )
    # Test fallback to float32 behavior if this machine doesn't support bfloat16
    assert model.model.dtype is torch.float32
    # Ensure that we can get something out of it
    pred = model.run("@bar what a cute cat!")
    assert isinstance(pred, GeneratedTextResult)


def test_prompt_output_types(causal_lm_train_kwargs):
    # Try training a model with output_model_types set to a list of strings
    patch_kwargs = {
        "num_epochs": 1,
        "verbalizer": "Tweet text : {{input}} Label : ",
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                caikit_nlp.data_model.GenerationTrainRecord(
                    input="@foo what a cute dog!", output="no complaint"
                ),
                caikit_nlp.data_model.GenerationTrainRecord(
                    input="@bar this is the worst idea ever.", output="complaint"
                ),
            ]
        ),
        "torch_dtype": torch.bfloat16,
        "device": "cpu",
        "tuning_config": caikit_nlp.data_model.TuningConfig(
            num_virtual_tokens=8,
            prompt_tuning_init_text="hello world",
            output_model_types=["DECODER"],
        ),
    }
    causal_lm_train_kwargs.update(patch_kwargs)
    model = caikit_nlp.modules.text_generation.PeftPromptTuning.train(
        **causal_lm_train_kwargs
    )
    assert model

    patch_kwargs = {
        "tuning_config": caikit_nlp.data_model.TuningConfig(
            num_virtual_tokens=8,
            prompt_tuning_init_text="hello world",
            output_model_types=[caikit_nlp.data_model.PromptOutputModelType.DECODER],
        )
    }
    model = caikit_nlp.modules.text_generation.PeftPromptTuning.train(
        **causal_lm_train_kwargs
    )
    assert model


def test_error_empty_stream(causal_lm_train_kwargs):
    patch_kwargs = {
        "num_epochs": 1,
        "verbalizer": "Tweet text : {{input}} Label : ",
        "train_stream": caikit.core.data_model.DataStream.from_iterable([]),
    }
    causal_lm_train_kwargs.update(patch_kwargs)
    with pytest.raises(ValueError):
        caikit_nlp.modules.text_generation.PeftPromptTuning.train(
            **causal_lm_train_kwargs
        )


### Implementation details
# These tests can probably be removed and tested directly through .save() once
# full seq2seq support is completed and verified.
def test_get_prompt_vector_encoder_only_seq2seq(seq2seq_lm_dummy_model):
    """Ensure that if we have an encoder/decoder PEFT model, we can get the encoder vectors."""
    peft_model = seq2seq_lm_dummy_model.model
    # Ensure we have 16 rows in our embed matrix; otherwise the below operations are invalid
    peft_model.peft_config[DEFAULT_ADAPTER].num_virtual_tokens = 16
    peft_model.peft_config[DEFAULT_ADAPTER].num_transformer_submodules = 1

    prompt_vectors = seq2seq_lm_dummy_model.get_exportable_prompt_vectors(
        peft_model,
        seq2seq_lm_dummy_model.tuning_type,
        seq2seq_lm_dummy_model.output_model_types,
    )
    # Get the dimensions of our word embeddings
    dummy_emb_input = torch.tensor([1]).to(peft_model.device)
    word_embedding_dim = peft_model.word_embeddings(dummy_emb_input).shape[1]
    # Ensure that only the encoder is of dim (num_virtual_tokens, word_embedding_dim)
    assert prompt_vectors[PeftPromptTuning._ENCODER_KEY.name].shape == (
        peft_model.peft_config[DEFAULT_ADAPTER].num_virtual_tokens,
        word_embedding_dim,
    )
    assert prompt_vectors[PeftPromptTuning._DECODER_KEY.name] is None


@pytest.mark.skip(
    """
We are skipping this test because currently we do not have
 support for encoder-decoder output for seq2seq
"""
)
def test_get_prompt_vector_encoder_decoder_seq2seq(seq2seq_lm_dummy_model):
    """Ensure that if we have an encoder/decoder PEFT model, we can get both prompt vectors."""
    peft_model = seq2seq_lm_dummy_model.model
    # Ensure we have 16 rows in our embed matrix; otherwise the below operations are invalid
    assert (
        peft_model.peft_config[DEFAULT_ADAPTER].num_virtual_tokens
        * peft_model.peft_config[DEFAULT_ADAPTER].num_transformer_submodules
        == 16
    )
    peft_model.peft_config[DEFAULT_ADAPTER].num_virtual_tokens = 8
    peft_model.peft_config[DEFAULT_ADAPTER].num_transformer_submodules = 2

    # Ensure that if we grab the prompt from the model, we have both an encoder and decoder.
    # So our embedding matrix should be (2 * num_virtual_tokens, word_embedding_dim)
    prompt_vectors = seq2seq_lm_dummy_model.get_exportable_prompt_vectors(
        peft_model,
        seq2seq_lm_dummy_model.tuning_type,
        seq2seq_lm_dummy_model.output_model_types,
    )
    # Get the dimensions of our word embeddings
    dummy_emb_input = torch.tensor([1]).to(peft_model.device)
    word_embedding_dim = peft_model.word_embeddings(dummy_emb_input).shape[1]

    # Ensure that the encoder / decoder are each of dim (num_virtual_tokens, word_embedding_dim)
    assert prompt_vectors[PeftPromptTuning._ENCODER_KEY.name].shape == (
        peft_model.peft_config[DEFAULT_ADAPTER].num_virtual_tokens,
        word_embedding_dim,
    )
    assert prompt_vectors[PeftPromptTuning._DECODER_KEY.name].shape == (
        peft_model.peft_config[DEFAULT_ADAPTER].num_virtual_tokens,
        word_embedding_dim,
    )


def test_model_can_only_have_one_or_two_transformer_modules(seq2seq_lm_dummy_model):
    """Ensure that we explode if we have more transformer modules than we should."""
    peft_model = seq2seq_lm_dummy_model.model
    # Ensure we have 16 rows in our embed matrix; otherwise the below operations are invalid
    assert (
        peft_model.peft_config[DEFAULT_ADAPTER].num_virtual_tokens
        * peft_model.peft_config[DEFAULT_ADAPTER].num_transformer_submodules
        == 16
    )
    # Peft doesn't enforce the range 1 <= x <= 2 at initialization time, so it's good to ensure the
    # export knows how to handle it directly.
    peft_model.peft_config[DEFAULT_ADAPTER].num_virtual_tokens = 4
    peft_model.peft_config[DEFAULT_ADAPTER].num_transformer_submodules = 4
    with pytest.raises(ValueError):
        PeftPromptTuning.get_exportable_prompt_vectors(
            peft_model,
            TuningType.PROMPT_TUNING,
            seq2seq_lm_dummy_model.output_model_types,
        )


######################## Test run with optional params #####################


def test_run_repetition_penalty_0_works(causal_lm_dummy_model):
    """Ensure repetition_penalty works with 0.0 as input"""
    pred = causal_lm_dummy_model.run("This text doesn't matter", repetition_penalty=0.0)
    assert isinstance(pred, GeneratedTextResult)


def test_run_truncate_tokens_0(causal_lm_dummy_model):
    """Ensure run function accepts 0 for truncation value
    and successfully turns off truncation"""
    pred = causal_lm_dummy_model.run(
        "This text doesn't matter", truncate_input_tokens=0
    )
    assert isinstance(pred, GeneratedTextResult)


def test_run_with_preserve_input_text(causal_lm_dummy_model):
    """Ensure preserve input text removes input
    from generated output when set to False"""
    input_text = "This text doesn't matter"
    pred = causal_lm_dummy_model.run(input_text, preserve_input_text=True)
    assert input_text in pred.generated_text
    pred = causal_lm_dummy_model.run(input_text, preserve_input_text=False)
    assert input_text not in pred.generated_text


def test_run_sampling_param_ignored_greedy_decoding(causal_lm_dummy_model):
    """Ensure sampling parameter gets ignored when decoding method
    is set to GREEDY
    """
    pred = causal_lm_dummy_model.run(
        "This text doesn't matter",
        decoding_method="GREEDY",
        top_k=2,
        top_p=0.23,
        typical_p=0.23,
        temperature=0.77,
    )
    assert isinstance(pred, GeneratedTextResult)


def test_run_with_custom_stop_criteria(causal_lm_dummy_model):
    """Ensure custom stop sequences works with run"""
    pred = causal_lm_dummy_model.run(
        "This text doesn't matter",
        decoding_method="GREEDY",
        stop_sequences=["Foo", "bar"],
    )
    assert isinstance(pred, GeneratedTextResult)


def test_run_exponential_decay_len_penatly_object(causal_lm_dummy_model):
    """Ensure exponential decay len penalty works with the data model
    object
    """
    penalty = ExponentialDecayLengthPenalty(1, 0.2)
    pred = causal_lm_dummy_model.run(
        "This text doesn't matter",
        decoding_method="GREEDY",
        stop_sequences=["Foo", "bar"],
        exponential_decay_length_penalty=penalty,
    )
    assert isinstance(pred, GeneratedTextResult)


def test_train_with_data_validation_raises(causal_lm_train_kwargs, set_cpu_device):
    """Check if we are able to throw error for when number of examples are more than configured limit"""
    patch_kwargs = {
        "num_epochs": 1,
        "verbalizer": "Tweet text : {{input}} Label : ",
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                ClassificationTrainRecord(
                    text="@foo what a cute dog!", labels=["no complaint"]
                ),
                ClassificationTrainRecord(
                    text="@bar this is the worst idea ever.", labels=["complaint"]
                ),
            ]
        ),
        "torch_dtype": torch.bfloat16,
        "device": "cpu",
    }
    causal_lm_train_kwargs.update(patch_kwargs)

    model_name = causal_lm_train_kwargs["base_model"]._model_name
    module = caikit_nlp.modules.text_generation.PeftPromptTuning
    with temp_config(training_data_limit={module.MODULE_ID: {model_name: 1}}):
        with pytest.raises(ValueError):
            module.train(**causal_lm_train_kwargs)


def test_train_with_data_validation_success(causal_lm_train_kwargs, set_cpu_device):
    """Check if we are able to train successfully if training data is within limits"""
    patch_kwargs = {
        "num_epochs": 1,
        "verbalizer": "Tweet text : {{input}} Label : ",
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                ClassificationTrainRecord(
                    text="@foo what a cute dog!", labels=["no complaint"]
                ),
                ClassificationTrainRecord(
                    text="@bar this is the worst idea ever.", labels=["complaint"]
                ),
            ]
        ),
        "torch_dtype": torch.bfloat16,
        "device": "cpu",
    }
    causal_lm_train_kwargs.update(patch_kwargs)

    model_name = causal_lm_train_kwargs["base_model"]._model_name
    module = caikit_nlp.modules.text_generation.PeftPromptTuning
    with temp_config(training_data_limit={module.MODULE_ID: {model_name: 2}}):

        model = module.train(**causal_lm_train_kwargs)
        assert model


def test_train_with_non_existent_limit_success(causal_lm_train_kwargs, set_cpu_device):
    """Check if we are able to train successfully if training data limit doesn't exist for particular model"""
    patch_kwargs = {
        "num_epochs": 1,
        "verbalizer": "Tweet text : {{input}} Label : ",
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                ClassificationTrainRecord(
                    text="@foo what a cute dog!", labels=["no complaint"]
                )
            ]
        ),
        "torch_dtype": torch.bfloat16,
        "device": "cpu",
    }
    causal_lm_train_kwargs.update(patch_kwargs)

    model_name = causal_lm_train_kwargs["base_model"]._model_name
    module = caikit_nlp.modules.text_generation.PeftPromptTuning
    with temp_config(training_data_limit={module.MODULE_ID: {"foo": 2}}):

        model = module.train(**causal_lm_train_kwargs)
        assert model


def test_train_with_no_limit_for_module(causal_lm_train_kwargs, set_cpu_device):
    """Check if we are able to train successfully if training data limit doesn't exist prompt tuning module"""
    patch_kwargs = {
        "num_epochs": 1,
        "verbalizer": "Tweet text : {{input}} Label : ",
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                ClassificationTrainRecord(
                    text="@foo what a cute dog!", labels=["no complaint"]
                )
            ]
        ),
        "torch_dtype": torch.bfloat16,
        "device": "cpu",
    }
    causal_lm_train_kwargs.update(patch_kwargs)

    model_name = causal_lm_train_kwargs["base_model"]._model_name
    module = caikit_nlp.modules.text_generation.PeftPromptTuning
    with temp_config(training_data_limit={}):

        model = module.train(**causal_lm_train_kwargs)
        assert model


def test_train_module_level_data_validation_raises(
    causal_lm_train_kwargs, set_cpu_device
):
    """Check if train raises with module level default configuration
    if training data is within limits and model config is not provided
    """
    patch_kwargs = {
        "num_epochs": 1,
        "verbalizer": "Tweet text : {{input}} Label : ",
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                ClassificationTrainRecord(
                    text="@foo what a cute dog!", labels=["no complaint"]
                ),
                ClassificationTrainRecord(
                    text="@bar this is the worst idea ever.", labels=["complaint"]
                ),
            ]
        ),
        "torch_dtype": torch.bfloat16,
        "device": "cpu",
    }
    causal_lm_train_kwargs.update(patch_kwargs)

    module = caikit_nlp.modules.text_generation.PeftPromptTuning
    with temp_config(
        training_data_limit={module.MODULE_ID: {"__default__": 1, "foo": 2}}
    ):
        with pytest.raises(ValueError):
            module.train(**causal_lm_train_kwargs)


def test_train_module_level_data_validation_success(
    causal_lm_train_kwargs, set_cpu_device
):
    """Check if we are able to train successfully with module level default configuration
    if training data is within limits and model config present
    """
    patch_kwargs = {
        "num_epochs": 1,
        "verbalizer": "Tweet text : {{input}} Label : ",
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                ClassificationTrainRecord(
                    text="@foo what a cute dog!", labels=["no complaint"]
                ),
                ClassificationTrainRecord(
                    text="@bar this is the worst idea ever.", labels=["complaint"]
                ),
            ]
        ),
        "torch_dtype": torch.bfloat16,
        "device": "cpu",
    }
    causal_lm_train_kwargs.update(patch_kwargs)

    model_name = causal_lm_train_kwargs["base_model"]._model_name
    module = caikit_nlp.modules.text_generation.PeftPromptTuning
    with temp_config(
        training_data_limit={module.MODULE_ID: {"__default__": 1, model_name: 2}}
    ):

        model = module.train(**causal_lm_train_kwargs)
        assert model


def test_train_global_default_data_validation_raises(
    causal_lm_train_kwargs, set_cpu_device
):
    """Check if train raises with global default configuration
    if training data is within limits and model config is not provided
    """
    patch_kwargs = {
        "num_epochs": 1,
        "verbalizer": "Tweet text : {{input}} Label : ",
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                ClassificationTrainRecord(
                    text="@foo what a cute dog!", labels=["no complaint"]
                ),
                ClassificationTrainRecord(
                    text="@bar this is the worst idea ever.", labels=["complaint"]
                ),
            ]
        ),
        "torch_dtype": torch.bfloat16,
        "device": "cpu",
    }
    causal_lm_train_kwargs.update(patch_kwargs)

    module = caikit_nlp.modules.text_generation.PeftPromptTuning
    with temp_config(
        training_data_limit={"__default__": 1, module.MODULE_ID: {"foo": 2}}
    ):
        with pytest.raises(ValueError):
            module.train(**causal_lm_train_kwargs)


def test_train_global_default_data_validation_success(
    causal_lm_train_kwargs, set_cpu_device
):
    """Check if we are able to train successfully with global default configuration
    if training data is within limits and model config is present
    """
    patch_kwargs = {
        "num_epochs": 1,
        "verbalizer": "Tweet text : {{input}} Label : ",
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                ClassificationTrainRecord(
                    text="@foo what a cute dog!", labels=["no complaint"]
                ),
                ClassificationTrainRecord(
                    text="@bar this is the worst idea ever.", labels=["complaint"]
                ),
            ]
        ),
        "torch_dtype": torch.bfloat16,
        "device": "cpu",
    }
    causal_lm_train_kwargs.update(patch_kwargs)

    model_name = causal_lm_train_kwargs["base_model"]._model_name
    module = caikit_nlp.modules.text_generation.PeftPromptTuning
    with temp_config(
        training_data_limit={"__default__": 1, module.MODULE_ID: {model_name: 2}}
    ):

        model = module.train(**causal_lm_train_kwargs)
        assert model
