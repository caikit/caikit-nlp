"""Tests for sequence classification module
"""
# Standard
import tempfile

# Third Party
from pytest import approx
import pytest

# First Party
from caikit.interfaces.nlp.data_model import ClassificationResult, ClassificationResults

# Local
from caikit_nlp.modules.text_classification import SequenceClassification
from tests.fixtures import SEQ_CLASS_MODEL

## Setup ########################################################################

# Bootstrapped sequence classification model for reusability across tests
# .bootstrap is tested separately in the first test
BOOTSTRAPPED_SEQ_CLASS_MODEL = SequenceClassification.bootstrap(SEQ_CLASS_MODEL)

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Once upon a time in a land far away",
]

## Tests ########################################################################
# Exact numbers here from the tiny model are not particularly important,
# but we check them here to make sure that the arrays are re-ordered correctly


def test_bootstrap_and_run():
    """Check if we can bootstrap and run sequence classification models"""
    model = SequenceClassification.bootstrap(SEQ_CLASS_MODEL)
    classification_result = model.run(TEXTS[0])
    assert isinstance(classification_result, ClassificationResults)
    assert len(classification_result.results) == 2  # 2 labels

    assert isinstance(classification_result.results[0], ClassificationResult)
    assert classification_result.results[0].label == "LABEL_0"
    assert approx(classification_result.results[0].score) == 0.49526197
    assert classification_result.results[1].label == "LABEL_1"
    assert approx(classification_result.results[1].score) == 0.50473803


def test_bootstrap_and_run_batch():
    """Check if we can bootstrap and run_batch sequence classification models"""
    classification_result_list = BOOTSTRAPPED_SEQ_CLASS_MODEL.run_batch(TEXTS)
    assert len(classification_result_list) == 2

    first_result = classification_result_list[0]
    assert isinstance(first_result, ClassificationResults)
    assert first_result.results[0].label == "LABEL_0"
    assert approx(first_result.results[0].score) == 0.49526197
    assert first_result.results[1].label == "LABEL_1"
    assert classification_result_list[1].results[0].label == "LABEL_0"


def test_load_save_and_run_model():
    """Check if we can load and run a saved model successfully"""
    with tempfile.TemporaryDirectory() as model_dir:
        BOOTSTRAPPED_SEQ_CLASS_MODEL.save(model_dir)
        new_model = SequenceClassification.load(model_dir)
        classification_result = new_model.run(TEXTS[0])
        assert isinstance(classification_result, ClassificationResults)
        assert len(classification_result.results) == 2  # 2 labels

        assert isinstance(classification_result.results[0], ClassificationResult)
        assert classification_result.results[0].label == "LABEL_0"
        assert approx(classification_result.results[0].score) == 0.49526197
