# Third Party
from transformers import Trainer
import pytest
import torch

# First Party
from caikit.interfaces.nlp.data_model import GeneratedTextResult
import caikit

# Local
from caikit_nlp.data_model import GenerationTrainRecord
from caikit_nlp.modules.text_generation import FineTuning
from caikit_nlp.resources.pretrained_model import HFAutoCausalLM, HFAutoSeq2SeqLM
from tests.fixtures import (
    CAUSAL_LM_MODEL,
    SEQ2SEQ_LM_MODEL,
    disable_wip,
    set_cpu_device,
)


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
    model = FineTuning.train(**train_kwargs)
    assert isinstance(model.model, HFAutoSeq2SeqLM)
    # Ensure that we can get something out of it
    pred = model.run("@bar what a cute cat!")
    assert isinstance(pred, GeneratedTextResult)


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
    model = FineTuning.train(**train_kwargs)
    assert isinstance(model.model, HFAutoCausalLM)

    # Ensure that we can get something out of it
    pred = model.run("@bar what a cute cat!")
    assert isinstance(pred, GeneratedTextResult)


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
    model = FineTuning.train(**train_kwargs)
    assert isinstance(model.model, Trainer)
