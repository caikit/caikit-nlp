"""Tests for text embedding module
"""
# Standard
import os
import tempfile

# Third Party
from pytest import approx
import numpy as np
import pytest

# Local
from caikit_nlp.data_model import EmbeddingResult
from caikit_nlp.modules.text_embedding import EmbeddingModule
from tests.fixtures import SEQ_CLASS_MODEL

## Setup ########################################################################

# Bootstrapped sequence classification model for reuse across tests
# .bootstrap is tested separately in the first test
BOOTSTRAPPED_MODEL = EmbeddingModule.bootstrap(SEQ_CLASS_MODEL)

INPUT = "The quick brown fox jumps over the lazy dog."

## Tests ########################################################################


def _assert_is_expected_embedding_result(actual):
    assert isinstance(actual, EmbeddingResult)
    assert isinstance(actual.result.data.values[0], np.float32)
    assert len(actual.result.data.values) == 32
    # Just testing a few values for readability
    assert approx(actual.result.data.values[0]) == 0.3244932293891907
    assert approx(actual.result.data.values[1]) == -0.4934631288051605
    assert approx(actual.result.data.values[2]) == 0.5721234083175659


def test_bootstrap_and_run():
    """Check if we can bootstrap and run embedding"""
    model = EmbeddingModule.bootstrap(SEQ_CLASS_MODEL)
    result = model.run(INPUT)
    _assert_is_expected_embedding_result(result)


def test_run_type_check():
    """Input cannot be a list"""
    model = BOOTSTRAPPED_MODEL
    with pytest.raises(TypeError):
        model.run([INPUT])
        pytest.fail("Should not reach here")


def test_save_load_and_run_model():
    """Check if we can load and run a saved model successfully"""
    model_id = "model_id"
    with tempfile.TemporaryDirectory(suffix="-1st") as model_dir:
        model_path = os.path.join(model_dir, model_id)
        BOOTSTRAPPED_MODEL.save(model_path)
        new_model = EmbeddingModule.load(model_path)

    result = new_model.run(input=INPUT)
    _assert_is_expected_embedding_result(result)


@pytest.mark.parametrize(
    "model_path", ["", " ", " " * 100], ids=["empty", "space", "spaces"]
)
def test_save_value_checks(model_path):
    with pytest.raises(ValueError):
        BOOTSTRAPPED_MODEL.save(model_path)


@pytest.mark.parametrize(
    "model_path",
    ["..", "../" * 100, "/", ".", " / ", " . "],
)
def test_save_exists_checks(model_path):
    """Tests for model paths are always existing dirs that should not be clobbered"""
    with pytest.raises(FileExistsError):
        BOOTSTRAPPED_MODEL.save(model_path)


def test_second_save_hits_exists_check():
    """Using a new path the first save should succeed but second fails"""
    model_id = "model_id"
    with tempfile.TemporaryDirectory(suffix="-2nd") as model_dir:
        model_path = os.path.join(model_dir, model_id)
        BOOTSTRAPPED_MODEL.save(model_path)
        with pytest.raises(FileExistsError):
            BOOTSTRAPPED_MODEL.save(model_path)


@pytest.mark.parametrize("model_path", [None, {}, object(), 1], ids=type)
def test_save_type_checks(model_path):
    with pytest.raises(TypeError):
        BOOTSTRAPPED_MODEL.save(model_path)
