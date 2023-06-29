"""Tests for sequence transformer sentence classification module
"""
# Standard
import os
import tempfile

# Third Party
from pytest import approx
import pytest

# First Party
from caikit.core.modules import ModuleBase, ModuleSaver, module

# Local
from caikit_nlp.data_model.classification import (
    TokenClassification,
    TokenClassificationResult,
)
from caikit_nlp.data_model.text import Span
from caikit_nlp.modules.text_classification import SequenceClassification
from caikit_nlp.modules.token_classification import TransformerSentenceClassification
from tests.fixtures import SEQ_CLASS_MODEL

## Setup ########################################################################

# Loaded sequence classification model for reusability across tests
LOADED_SEQ_CLASS_MODEL = SequenceClassification.load(SEQ_CLASS_MODEL)

DOCUMENT = (
    "The quick brown fox jumps over the lazy dog. Once upon a time in a land far away"
)

# Sentence splitter for tests
@module("4c9387f9-3683-4a94-bed9-8ecc1bf3ce47", "FakeTestSentenceSplitter", "0.0.1")
class FakeTestSentenceSplitter(ModuleBase):
    def run(self, text: str):
        return [
            Span(start=0, end=44, text="The quick brown fox jumps over the lazy dog."),
            Span(start=45, end=80, text="Once upon a time in a land far away"),
        ]

    def save(self, model_path: str):
        module_saver = ModuleSaver(
            self,
            model_path=model_path,
        )
        with module_saver:
            module_saver.update_config({})

    @classmethod
    def load(cls, model_path: str):
        return FakeTestSentenceSplitter()


SENTENCE_SPLITTER = FakeTestSentenceSplitter()

## Tests ########################################################################


def test_init_run():
    """Check if we can init and run sentence classification models with min arguments"""
    model = TransformerSentenceClassification(
        lang="en",
        sentence_splitter=SENTENCE_SPLITTER,
        sequence_classifier=LOADED_SEQ_CLASS_MODEL,
        default_threshold=0.5,
    )
    token_classification_result = model.run(DOCUMENT)
    assert isinstance(token_classification_result, TokenClassificationResult)
    assert len(token_classification_result.results) == 2  # 2 results over 0.5 expected
    assert isinstance(token_classification_result.results[0], TokenClassification)
    first_result = token_classification_result.results[0]
    assert first_result.start == 0
    assert first_result.end == 44
    assert first_result.word == "The quick brown fox jumps over the lazy dog."
    assert first_result.entity == "LABEL_1"
    assert approx(first_result.score) == 0.50473803
    assert token_classification_result.results[1].entity == "LABEL_1"


def test_init_run_with_threshold():
    """Check if we can run sentence classification models with overriden threshold"""
    model = TransformerSentenceClassification(
        lang="en",
        sentence_splitter=SENTENCE_SPLITTER,
        sequence_classifier=LOADED_SEQ_CLASS_MODEL,
        default_threshold=0.5,
    )
    token_classification_result = model.run(DOCUMENT, threshold=0.0)
    assert isinstance(token_classification_result, TokenClassificationResult)
    assert (
        len(token_classification_result.results) == 4
    )  # 4 (all) results over 0.0 expected


def test_init_run_with_optional_labels_to_output():
    """Check if we can run sentence classification models with labels_to_output"""
    model = TransformerSentenceClassification(
        lang="en",
        sentence_splitter=SENTENCE_SPLITTER,
        sequence_classifier=LOADED_SEQ_CLASS_MODEL,
        default_threshold=0.5,
        labels_to_output=["LABEL_0"],
    )
    token_classification_result = model.run(DOCUMENT, threshold=0.0)
    # All results would be above threshold 0.0 but only return those corresponding to label
    assert len(token_classification_result.results) == 2
    first_result = token_classification_result.results[0]
    assert first_result.start == 0
    assert first_result.end == 44
    assert first_result.word == "The quick brown fox jumps over the lazy dog."
    assert first_result.entity == "LABEL_0"
    assert approx(first_result.score) == 0.49526197


def test_init_with_optional_labels_mapping():
    """Check if we can run sentence classification models with labels_mapping"""
    model = TransformerSentenceClassification(
        lang="en",
        sentence_splitter=SENTENCE_SPLITTER,
        sequence_classifier=LOADED_SEQ_CLASS_MODEL,
        default_threshold=0.5,
        labels_mapping={"LABEL_0": "YAY", "LABEL_1": "NAY"},
    )
    token_classification_result = model.run(DOCUMENT, threshold=0.0)
    assert len(token_classification_result.results) == 4
    assert token_classification_result.results[0].entity == "YAY"
    assert approx(token_classification_result.results[0].score) == 0.49526197
    assert token_classification_result.results[3].entity == "NAY"
    assert approx(token_classification_result.results[3].score) == 0.50475168


def test_save_load_and_run_model():
    """Check if we can run a saved model successfully"""
    model = TransformerSentenceClassification(
        lang="en",
        sentence_splitter=SENTENCE_SPLITTER,
        sequence_classifier=LOADED_SEQ_CLASS_MODEL,
        default_threshold=0.5,
    )
    with tempfile.TemporaryDirectory() as model_dir:
        model.save(model_dir)
        assert os.path.exists(os.path.join(model_dir, "config.yml"))
        assert os.path.exists(os.path.join(model_dir, "sentence_split"))
        assert os.path.exists(os.path.join(model_dir, "sequence_classification"))

        new_model = TransformerSentenceClassification.load(model_dir)
        token_classification_result = new_model.run(DOCUMENT)
        assert isinstance(token_classification_result, TokenClassificationResult)
        assert (
            len(token_classification_result.results) == 2
        )  # 2 results over 0.5 expected
