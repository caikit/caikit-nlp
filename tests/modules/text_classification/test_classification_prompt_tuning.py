"""Tests for sequence classification module
"""

# Third Party
import torch

# First Party
from caikit.interfaces.nlp.data_model import ClassificationTrainRecord
import caikit

# Local
from caikit_nlp.modules.text_classification.classification_prompt_tuning import ClassificationPeftPromptTuning
from tests.fixtures import (
    causal_lm_dummy_model,
    causal_lm_train_kwargs,
    seq2seq_lm_dummy_model,
    seq2seq_lm_train_kwargs,
    set_cpu_device,
)

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
    model = ClassificationPeftPromptTuning.train(
        **causal_lm_train_kwargs
    )
    # Test fallback to float32 behavior if this machine doesn't support bfloat16
    assert model.classifier.model.dtype is torch.float32