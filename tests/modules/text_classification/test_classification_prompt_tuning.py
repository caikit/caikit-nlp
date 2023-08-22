"""Tests for sequence classification module
"""
# Standard
import os
import tempfile

# Third Party
import torch

# First Party
from caikit.interfaces.nlp.data_model import ClassificationTrainRecord
import caikit

# Local
from caikit_nlp.modules.text_classification.classification_prompt_tuning import (
    ClassificationPeftPromptTuning,
)
from caikit_nlp.modules.text_generation.peft_prompt_tuning import PeftPromptTuning
from tests.fixtures import (
    causal_lm_dummy_model,
    causal_lm_train_kwargs,
    seq2seq_lm_dummy_model,
    seq2seq_lm_train_kwargs,
    set_cpu_device,
)
import caikit_nlp


def test_train_model_classification_record(causal_lm_train_kwargs):
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
    model = ClassificationPeftPromptTuning.train(**causal_lm_train_kwargs)
    # Test fallback to float32 behavior if this machine doesn't support bfloat16
    assert model.classifier.model.dtype is torch.float32


####################
## save/load(...) ##
####################


def test_save(causal_lm_dummy_model):
    classifier_model = ClassificationPeftPromptTuning(
        classifier=causal_lm_dummy_model, unique_class_labels=["label1", "label2"]
    )
    # with tempfile.TemporaryDirectory() as model_dir:
    model_dir = "example_model_2"
    classifier_model.save(model_dir)
    assert os.path.exists(os.path.join(model_dir, "config.yml"))
    assert os.path.exists(os.path.join(model_dir, "artifacts", "config.yml"))


def test_save_and_load(causal_lm_dummy_model):
    classifier_model = ClassificationPeftPromptTuning(
        classifier=causal_lm_dummy_model, unique_class_labels=["label1", "label2"]
    )
    with tempfile.TemporaryDirectory() as model_dir:
        classifier_model.save(model_dir)
        model_load = caikit_nlp.load(model_dir)
        assert isinstance(model_load, ClassificationPeftPromptTuning)
        assert isinstance(model_load.classifier, PeftPromptTuning)
        assert model_load.unique_class_labels == ["label1", "label2"]
