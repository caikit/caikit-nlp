"""Tests for filtered span classification module
"""
# Standard
from typing import List
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
from caikit_nlp.modules.token_classification import (
    FilteredSpanClassification,
    TokenClassificationTask,
)
from tests.fixtures import SEQ_CLASS_MODEL

## Setup ########################################################################

# Bootstrapped sequence classification model for reusability across tests
BOOTSTRAPPED_SEQ_CLASS_MODEL = SequenceClassification.bootstrap(SEQ_CLASS_MODEL)

DOCUMENT = (
    "The quick brown fox jumps over the lazy dog. Once upon a time in a land far away"
)

# Span/sentence splitter for tests
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

# Module that already returns token classification for tests
@module(
    "44d61711-c64b-4774-a39f-a9f40f1fcff0",
    "FakeTokenClassificationModule",
    "0.0.1",
    task=TokenClassificationTask,
)
class FakeTokenClassificationModule(ModuleBase):
    def run(self, text: str) -> TokenClassificationResult:
        pass

    def run_batch(self, texts: List[str]) -> List[TokenClassificationResult]:
        return [
            TokenClassificationResult(
                results=[
                    TokenClassification(
                        start=7, end=12, word="goose", entity="animal", score=0.3
                    ),
                    TokenClassification(
                        start=0, end=5, word="moose", entity="animal", score=0.8
                    ),
                ]
            ),
            TokenClassificationResult(
                results=[
                    TokenClassification(
                        start=0, end=4, word="iris", entity="plant", score=0.7
                    )
                ]
            ),
        ]


TOKEN_CLASSIFICATION_MODULE = FakeTokenClassificationModule()

## Tests ########################################################################


def test_bootstrap_run():
    """Check if we can bootstrap and run span classification models with min arguments"""
    model = FilteredSpanClassification.bootstrap(
        lang="en",
        span_splitter=SENTENCE_SPLITTER,
        classifier=BOOTSTRAPPED_SEQ_CLASS_MODEL,
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


def test_bootstrap_run_with_threshold():
    """Check if we can bootstrap span classification models with overriden threshold"""
    model = FilteredSpanClassification.bootstrap(
        lang="en",
        span_splitter=SENTENCE_SPLITTER,
        classifier=BOOTSTRAPPED_SEQ_CLASS_MODEL,
        default_threshold=0.5,
    )
    token_classification_result = model.run(DOCUMENT, threshold=0.0)
    assert isinstance(token_classification_result, TokenClassificationResult)
    assert (
        len(token_classification_result.results) == 4
    )  # 4 (all) results over 0.0 expected


def test_bootstrap_run_with_optional_labels_to_output():
    """Check if we can run span classification models with labels_to_output"""
    model = FilteredSpanClassification.bootstrap(
        lang="en",
        span_splitter=SENTENCE_SPLITTER,
        classifier=BOOTSTRAPPED_SEQ_CLASS_MODEL,
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


def test_bootstrap_run_with_token_classification():
    """Check if we can run span classification models with classifier that does token classification"""
    model = FilteredSpanClassification.bootstrap(
        lang="en",
        span_splitter=SENTENCE_SPLITTER,
        classifier=TOKEN_CLASSIFICATION_MODULE,
        default_threshold=0.5,
    )
    token_classification_result = model.run(DOCUMENT)
    assert isinstance(token_classification_result, TokenClassificationResult)
    assert len(token_classification_result.results) == 2  # 2 results over 0.5 expected
    assert isinstance(token_classification_result.results[0], TokenClassification)
    first_result = token_classification_result.results[0]
    assert first_result.start == 0
    assert first_result.end == 5
    assert first_result.word == "moose"
    assert first_result.entity == "animal"
    assert first_result.score == 0.8


def test_save_load_and_run_model():
    """Check if we can run a saved model successfully"""
    model = FilteredSpanClassification.bootstrap(
        lang="en",
        span_splitter=SENTENCE_SPLITTER,
        classifier=BOOTSTRAPPED_SEQ_CLASS_MODEL,
        default_threshold=0.5,
    )
    with tempfile.TemporaryDirectory() as model_dir:
        model.save(model_dir)
        assert os.path.exists(os.path.join(model_dir, "config.yml"))
        assert os.path.exists(os.path.join(model_dir, "span_split"))
        assert os.path.exists(os.path.join(model_dir, "sequence_classification"))

        new_model = FilteredSpanClassification.load(model_dir)
        token_classification_result = new_model.run(DOCUMENT)
        assert isinstance(token_classification_result, TokenClassificationResult)
        assert (
            len(token_classification_result.results) == 2
        )  # 2 results over 0.5 expected
