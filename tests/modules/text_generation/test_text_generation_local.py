"""Tests for text-generation module
"""
# Standard
import os
import platform
import tempfile

# Third Party
import pytest
import torch

# First Party
from caikit.interfaces.nlp.data_model import GeneratedTextResult
import caikit

# Local
from caikit_nlp.data_model import GenerationTrainRecord
from caikit_nlp.modules.text_generation import TextGeneration
from caikit_nlp.resources.pretrained_model import HFAutoCausalLM, HFAutoSeq2SeqLM
from tests.fixtures import (
    CAUSAL_LM_MODEL,
    SEQ2SEQ_LM_MODEL,
    disable_wip,
    set_cpu_device,
)

### Stub Modules


def test_bootstrap_and_run_causallm():
    """Check if we can bootstrap and run causallm models"""

    model = TextGeneration.bootstrap(CAUSAL_LM_MODEL)

    sample_text = "Hello stub"
    generated_text = model.run(sample_text)
    assert isinstance(generated_text, GeneratedTextResult)


def test_bootstrap_and_run_seq2seq():
    """Check if we can bootstrap and run seq2seq models"""

    model = TextGeneration.bootstrap(SEQ2SEQ_LM_MODEL)

    sample_text = "Hello stub"
    generated_text = model.run(sample_text)
    assert isinstance(generated_text, GeneratedTextResult)


def test_bootstrap_and_save_model():
    """Check if we can bootstrap and save the model successfully"""

    model = TextGeneration.bootstrap(SEQ2SEQ_LM_MODEL)

    with tempfile.TemporaryDirectory() as model_dir:
        model.save(model_dir)
        assert os.path.isfile(os.path.join(model_dir, "config.yml"))


def test_save_model_can_run():
    """Check if the model we bootstrap and save is able to load and run successfully"""
    model = TextGeneration.bootstrap(SEQ2SEQ_LM_MODEL)

    with tempfile.TemporaryDirectory() as model_dir:
        model.save(model_dir)
        del model
        new_model = TextGeneration.load(model_dir)
        sample_text = "Hello stub"
        generated_text = new_model.run(sample_text)
        assert isinstance(generated_text, GeneratedTextResult)


############################## Training ################################


@pytest.mark.skipif(platform.processor() == "arm", reason="ARM training not supported")
def test_train_model_seq2seq(disable_wip, set_cpu_device):
    """Ensure that we can finetune a seq2seq model on some toy data for 1+
    steps & run inference."""
    train_kwargs = {
        "base_model": HFAutoSeq2SeqLM.bootstrap(
            model_name=SEQ2SEQ_LM_MODEL, tokenizer_name=SEQ2SEQ_LM_MODEL
        ),
        "num_epochs": 1,
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                GenerationTrainRecord(
                    input="@foo what a cute dog!", output="no complaint"
                ),
                GenerationTrainRecord(
                    input="@bar this is the worst idea ever.", output="complaint"
                ),
            ]
        ),
        "torch_dtype": torch.float32,
    }
    model = TextGeneration.train(**train_kwargs)
    assert isinstance(model.model, HFAutoSeq2SeqLM)

    # Ensure that we can get something out of it
    pred = model.run("@bar what a cute cat!")
    assert isinstance(pred, GeneratedTextResult)


@pytest.mark.skipif(platform.processor() == "arm", reason="ARM training not supported")
def test_train_model_save_and_load(disable_wip, set_cpu_device):
    """Ensure that we are able to save and load a finetuned model and execute inference on it"""
    train_kwargs = {
        "base_model": HFAutoSeq2SeqLM.bootstrap(
            model_name=SEQ2SEQ_LM_MODEL, tokenizer_name=SEQ2SEQ_LM_MODEL
        ),
        "num_epochs": 1,
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                GenerationTrainRecord(
                    input="@foo what a cute dog!", output="no complaint"
                )
            ]
        ),
        "torch_dtype": torch.float32,
    }
    model = TextGeneration.train(**train_kwargs)
    assert isinstance(model.model, HFAutoSeq2SeqLM)
    with tempfile.TemporaryDirectory() as model_dir:
        model.save(model_dir)
        new_model = TextGeneration.load(model_dir)
        sample_text = "Hello stub"
        generated_text = new_model.run(sample_text)
        assert isinstance(generated_text, GeneratedTextResult)


@pytest.mark.skipif(platform.processor() == "arm", reason="ARM training not supported")
def test_train_model_causallm(disable_wip, set_cpu_device):
    """Ensure that we can finetune a causal-lm model on some toy data for 1+
    steps & run inference."""
    train_kwargs = {
        "base_model": HFAutoCausalLM.bootstrap(
            model_name=CAUSAL_LM_MODEL, tokenizer_name=CAUSAL_LM_MODEL
        ),
        "num_epochs": 1,
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                GenerationTrainRecord(
                    input="@foo what a cute dog!", output="no complaint"
                ),
            ]
        ),
        "torch_dtype": torch.float32,
    }
    model = TextGeneration.train(**train_kwargs)
    assert isinstance(model.model, HFAutoCausalLM)

    # Ensure that we can get something out of it
    pred = model.run("@bar what a cute cat!")
    assert isinstance(pred, GeneratedTextResult)


############################## Inferencing flags ################################


@pytest.mark.skipif(platform.processor() == "arm", reason="ARM training not supported")
def test_train_model_causallm(disable_wip, set_cpu_device):
    """Ensure that we can finetune a causal-lm model on some toy data for 1+
    steps & run inference."""
    train_kwargs = {
        "base_model": HFAutoCausalLM.bootstrap(
            model_name=CAUSAL_LM_MODEL, tokenizer_name=CAUSAL_LM_MODEL
        ),
        "num_epochs": 1,
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                GenerationTrainRecord(
                    input="@foo what a cute dog!", output="no complaint"
                ),
            ]
        ),
        "torch_dtype": torch.float32,
    }
    model = TextGeneration.train(**train_kwargs)
    assert isinstance(model.model, HFAutoCausalLM)

    # Ensure that preserve_input_text returns input in output
    pred = model.run("@bar what a cute cat!", preserve_input_text=True)
    assert "@bar what a cute cat!" in pred.generated_text

    # Ensure that preserve_input_text set to False, removes input from output
    pred = model.run("@bar what a cute cat!", preserve_input_text=False)
    assert "@bar what a cute cat!" not in pred.generated_text


############################## Error Cases ################################


def test_zero_epoch_case(disable_wip):
    """Test to ensure 0 epoch training request doesn't explode"""
    train_kwargs = {
        "base_model": HFAutoSeq2SeqLM.bootstrap(
            model_name=SEQ2SEQ_LM_MODEL, tokenizer_name=SEQ2SEQ_LM_MODEL
        ),
        "num_epochs": 0,
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                GenerationTrainRecord(
                    input="@foo what a cute dog!", output="no complaint"
                ),
            ]
        ),
        "torch_dtype": torch.float32,
    }
    model = TextGeneration.train(**train_kwargs)
    assert isinstance(model.model, HFAutoSeq2SeqLM)
