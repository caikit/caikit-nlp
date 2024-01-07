"""Tests for sequence classification module
"""
# Standard
import os
import tempfile

# Third Party
import pytest
import torch

# First Party
from caikit.interfaces.nlp.data_model import (
    ClassificationResults,
    ClassificationTrainRecord,
)
import caikit

# Local
from caikit_nlp.modules.text_classification.classification_prompt_tuning import (
    ClassificationPeftPromptTuning,
)
from caikit_nlp.modules.text_generation.peft_prompt_tuning import PeftPromptTuning
from tests.fixtures import causal_lm_dummy_model, causal_lm_train_kwargs
import caikit_nlp

####################
## train/run      ##
####################


def test_train_model(causal_lm_train_kwargs):
    """Ensure that we can train a model on some toy data for 1+ steps"""
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
    assert isinstance(model, ClassificationPeftPromptTuning)


# TODO: add test for scores in future when implemented
def test_run_classification_model(causal_lm_dummy_model):
    classifier_model = ClassificationPeftPromptTuning(
        classifier=causal_lm_dummy_model,
        unique_class_labels=["LABEL_0", "LABEL_1", "LABEL_2"],
    )
    output = classifier_model.run("Text does not matter")
    assert isinstance(output, ClassificationResults)
    # Returns supported class labels or None
    classifier_model.unique_class_labels.append(None)
    assert output.results[0].label in classifier_model.unique_class_labels
    assert output.results[0].score == None


def test_train_run_model_classification_record(causal_lm_train_kwargs):
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
    assert isinstance(model, ClassificationPeftPromptTuning)
    output = model.run("Text does not matter")
    assert isinstance(output, ClassificationResults)
    assert model.unique_class_labels == ["complaint", "no complaint"]
    # Returns supported class labels or None
    model.unique_class_labels.append(None)
    assert output.results[0].label in model.unique_class_labels
    assert output.results[0].score == None


####################
## save/load      ##
####################


def test_save(causal_lm_dummy_model):
    classifier_model = ClassificationPeftPromptTuning(
        classifier=causal_lm_dummy_model, unique_class_labels=["label1", "label2"]
    )
    with tempfile.TemporaryDirectory() as model_dir:
        classifier_model.save(model_dir)
        assert os.path.exists(os.path.join(model_dir, "config.yml"))
        assert os.path.exists(os.path.join(model_dir, "artifacts", "config.yml"))


def test_save_and_reload_with_base_model(causal_lm_dummy_model):
    classifier_model = ClassificationPeftPromptTuning(
        classifier=causal_lm_dummy_model, unique_class_labels=["label1", "label2"]
    )
    with tempfile.TemporaryDirectory() as model_dir:
        classifier_model.save(model_dir, save_base_model=True)
        model_load = caikit_nlp.load(model_dir)
        assert isinstance(model_load, ClassificationPeftPromptTuning)
        assert isinstance(model_load.classifier, PeftPromptTuning)
        assert model_load.unique_class_labels == ["label1", "label2"]


def test_save_and_reload_without_base_model(causal_lm_dummy_model):
    """Ensure that if we don't save the base model, we get the expected behavior."""
    with tempfile.TemporaryDirectory() as model_dir:
        causal_lm_dummy_model.save(model_dir, save_base_model=False)
        # For now, if we are missing the base model at load time, we throw ValueError
        with pytest.raises(ValueError):
            caikit_nlp.load(model_dir)


####################
## save/load/run  ##
####################


def test_save_reload_and_run_with_base_model(causal_lm_dummy_model):
    classifier_model = ClassificationPeftPromptTuning(
        classifier=causal_lm_dummy_model, unique_class_labels=["label1", "label2"]
    )
    with tempfile.TemporaryDirectory() as model_dir:
        classifier_model.save(model_dir, save_base_model=True)
        model_load = caikit_nlp.load(model_dir)
    assert isinstance(model_load, ClassificationPeftPromptTuning)
    assert isinstance(model_load.classifier, PeftPromptTuning)
    assert model_load.unique_class_labels == ["label1", "label2"]
    output = model_load.run("Text does not matter")
    assert isinstance(output, ClassificationResults)
    # Returns supported class labels or None
    model_load.unique_class_labels.append(None)
    assert output.results[0].label in model_load.unique_class_labels
    assert output.results[0].score == None
