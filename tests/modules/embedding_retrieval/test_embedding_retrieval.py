"""Tests for sequence classification module
"""
# Standard
import tempfile

# Third Party
from pytest import approx

# Local
from caikit_nlp.data_model import EmbeddingResult, Vector1D
from caikit_nlp.modules.embedding_retrieval import EmbeddingModule
from tests.fixtures import SEQ_CLASS_MODEL

## Setup ########################################################################

# Bootstrapped sequence classification model for reusability across tests
# .bootstrap is tested separately in the first test
BOOTSTRAPPED_MODEL = EmbeddingModule.bootstrap(SEQ_CLASS_MODEL)

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Once upon a time in a land far away",
]

## Tests ########################################################################
# Exact numbers here from the tiny model are not particularly important,
# but we check them here to make sure that the arrays are re-ordered correctly


def test_bootstrap_and_run_list():
    """Check if we can bootstrap and run embedding"""
    model = EmbeddingModule.bootstrap(SEQ_CLASS_MODEL)
    embedding_result = model.run(TEXTS)

    assert isinstance(embedding_result, EmbeddingResult)
    assert (
        len(embedding_result.data) == 2 == len(TEXTS)
    )  # 2 vectors for 2 input sentences
    assert isinstance(embedding_result.data[0], Vector1D)
    assert len(embedding_result.data[0].data) == 32
    assert embedding_result.data[0].data[0] == 0.3244932293891907
    assert embedding_result.data[1].data[1] == -0.3782769441604614
    assert approx(embedding_result.data[1].data[2]) == 0.7745956


def test_bootstrap_and_run_str():
    """Check if we can bootstrap and run when given a string as input"""
    model = BOOTSTRAPPED_MODEL
    embedding_result = model.run(TEXTS[0])  # string input will be converted to list
    assert isinstance(embedding_result, EmbeddingResult)
    assert len(embedding_result.data) == 1  # 1 vector for one input sentence
    assert isinstance(embedding_result.data[0], Vector1D)
    assert approx(embedding_result.data[0].data[0]) == 0.32449323


def test_load_save_and_run_model():
    """Check if we can load and run a saved model successfully"""
    with tempfile.TemporaryDirectory() as model_dir:
        BOOTSTRAPPED_MODEL.save(model_dir)
        new_model = EmbeddingModule.load(model_dir)

    embedding_result = new_model.run(input=TEXTS)
    assert isinstance(embedding_result, EmbeddingResult)
    assert (
        len(embedding_result.data) == 2 == len(TEXTS)
    )  # 2 vectors for 2 input sentences
    assert isinstance(embedding_result.data[0], Vector1D)
    assert len(embedding_result.data[0].data) == 32
    assert embedding_result.data[0].data[0] == 0.3244932293891907
    assert embedding_result.data[1].data[1] == -0.3782769441604614
    assert approx(embedding_result.data[1].data[2]) == 0.7745956
