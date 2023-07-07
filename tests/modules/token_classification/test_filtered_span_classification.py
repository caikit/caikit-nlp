"""Tests for filtered span classification module
"""
# Standard
from typing import Iterable, List
import os
import tempfile

# Third Party
from pytest import approx
import pytest

# First Party
from caikit.core import data_model
from caikit.core.modules import ModuleBase, module

# Local
from caikit_nlp.data_model.classification import (
    TokenClassification,
    TokenClassificationResult,
)
from caikit_nlp.modules.text_classification import SequenceClassification
from caikit_nlp.modules.token_classification import (
    FilteredSpanClassification,
    TokenClassificationTask,
)
from caikit_nlp.modules.tokenization.regex_sentence_splitter import (
    RegexSentenceSplitter,
)
from tests.fixtures import SEQ_CLASS_MODEL

## Setup ########################################################################

# Bootstrapped sequence classification model
BOOTSTRAPPED_SEQ_CLASS_MODEL = SequenceClassification.bootstrap(SEQ_CLASS_MODEL)
# Regex sentence splitter model
SENTENCE_TOKENIZER = RegexSentenceSplitter.bootstrap(
    "[^.!?\s][^.!?\n]*(?:[.!?](?!['\"]?\s|$)[^.!?]*)*[.!?]?['\"]?(?=\s|$)"
)

DOCUMENT = (
    "The quick brown fox jumps over the lazy dog. Once upon a time in a land far away."
)

# Token classifications in document
FOX_CLASS = TokenClassification(
    start=16, end=19, word="fox", entity="animal", score=0.8
)
DOG_CLASS = TokenClassification(
    start=40, end=43, word="dog", entity="animal", score=0.3
)
LAND_CLASS = TokenClassification(
    start=22, end=26, word="land", entity="thing", score=0.7
)
TOK_CLASSIFICATION_RESULT = TokenClassificationResult(results=[FOX_CLASS, DOG_CLASS])

# Modules that already returns token classification for tests
@module(
    "44d61711-c64b-4774-a39f-a9f40f1fcff0",
    "FakeTokenClassificationModule",
    "0.0.1",
    task=TokenClassificationTask,
)
class FakeTokenClassificationModule(ModuleBase):
    # This returns results for the whole document
    def run(self, text: str) -> TokenClassificationResult:
        return TOK_CLASSIFICATION_RESULT

    def run_batch(self, texts: List[str]) -> List[TokenClassificationResult]:
        return [
            TOK_CLASSIFICATION_RESULT,
            TokenClassificationResult(results=[LAND_CLASS]),
        ]


class StreamFakeTokenClassificationModule(FakeTokenClassificationModule):
    # Make module return results per sentence
    def run(self, text: str) -> TokenClassificationResult:
        if "land" in text:
            return TokenClassificationResult(results=[LAND_CLASS])
        else:
            return TOK_CLASSIFICATION_RESULT


class EmptyResFakeTokenClassificationModule(FakeTokenClassificationModule):
    def run(self, text: str) -> TokenClassificationResult:
        return TokenClassificationResult(results=[])

    def run_batch(self, texts: List[str]) -> List[TokenClassificationResult]:
        return [
            TokenClassificationResult(results=[]),
            TokenClassificationResult(results=[]),
        ]


TOKEN_CLASSIFICATION_MODULE = FakeTokenClassificationModule()
STREAM_TOKEN_CLASSIFICATION_MODULE = StreamFakeTokenClassificationModule()
EMPTY_RES_TOKEN_CLASSIFICATION_MODULE = EmptyResFakeTokenClassificationModule()

## Tests ########################################################################


def test_bootstrap_run():
    """Check if we can bootstrap and run span classification models with min arguments"""
    model = FilteredSpanClassification.bootstrap(
        lang="en",
        tokenizer=SENTENCE_TOKENIZER,
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
        tokenizer=SENTENCE_TOKENIZER,
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
        tokenizer=SENTENCE_TOKENIZER,
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
        tokenizer=SENTENCE_TOKENIZER,
        classifier=TOKEN_CLASSIFICATION_MODULE,
        default_threshold=0.5,
    )
    token_classification_result = model.run(DOCUMENT)
    assert isinstance(token_classification_result, TokenClassificationResult)
    assert len(token_classification_result.results) == 2  # 2 results over 0.5 expected
    assert isinstance(token_classification_result.results[0], TokenClassification)
    first_result = token_classification_result.results[0]
    assert first_result.start == 16
    assert first_result.end == 19
    assert first_result.word == "fox"
    assert first_result.entity == "animal"
    assert first_result.score == 0.8


def test_bootstrap_run_with_token_classification_no_results():
    """Check if we can run span classification models with classifier that does token classification
    but returns no results"""
    model = FilteredSpanClassification.bootstrap(
        lang="en",
        tokenizer=SENTENCE_TOKENIZER,
        classifier=EMPTY_RES_TOKEN_CLASSIFICATION_MODULE,
        default_threshold=0.5,
    )
    token_classification_result = model.run(DOCUMENT)
    assert isinstance(token_classification_result, TokenClassificationResult)
    assert len(token_classification_result.results) == 0


def test_save_load_and_run_model():
    """Check if we can run a saved model successfully"""
    model = FilteredSpanClassification.bootstrap(
        lang="en",
        tokenizer=SENTENCE_TOKENIZER,
        classifier=BOOTSTRAPPED_SEQ_CLASS_MODEL,
        default_threshold=0.5,
    )
    with tempfile.TemporaryDirectory() as model_dir:
        model.save(model_dir)
        assert os.path.exists(os.path.join(model_dir, "config.yml"))
        assert os.path.exists(os.path.join(model_dir, "tokenizer"))
        assert os.path.exists(os.path.join(model_dir, "classification"))

        new_model = FilteredSpanClassification.load(model_dir)
        token_classification_result = new_model.run(DOCUMENT)
        assert isinstance(token_classification_result, TokenClassificationResult)
        assert (
            len(token_classification_result.results) == 2
        )  # 2 results over 0.5 expected


### Streaming tests ##############################################################


def test_run_bidi_stream_model():
    """Check if model prediction works as expected for bi-directional stream"""

    stream_input = data_model.DataStream.from_iterable(DOCUMENT)
    model = FilteredSpanClassification.bootstrap(
        lang="en",
        tokenizer=SENTENCE_TOKENIZER,
        classifier=BOOTSTRAPPED_SEQ_CLASS_MODEL,
        default_threshold=0.5,
    )

    streaming_token_classification_result = model.run_bidi_stream(stream_input)
    assert isinstance(streaming_token_classification_result, Iterable)
    # Convert to list to more easily check outputs
    result_list = list(streaming_token_classification_result)

    first_result = result_list[0].results[0]
    assert isinstance(first_result, TokenClassification)
    assert first_result.start == 0
    assert first_result.end == 44
    assert first_result.word == "The quick brown fox jumps over the lazy dog."
    assert first_result.entity == "LABEL_1"
    assert approx(first_result.score) == 0.50473803

    # Check processed indices
    assert result_list[0].processed_index == 44
    assert result_list[1].processed_index == len(stream_input)

    # Assert total number of results should be equal to expected number of sentences
    expected_number_of_sentences = 2  # Sentence tokenizer returns 2 results
    count = len(result_list)
    assert count == expected_number_of_sentences


def test_run_bidi_stream_with_token_classification():
    """Check if model prediction with token classification
    works as expected for bi-directional stream"""

    stream_input = data_model.DataStream.from_iterable(DOCUMENT)
    model = FilteredSpanClassification.bootstrap(
        lang="en",
        tokenizer=SENTENCE_TOKENIZER,
        classifier=STREAM_TOKEN_CLASSIFICATION_MODULE,
        default_threshold=0.3,
    )
    streaming_token_classification_result = model.run_bidi_stream(stream_input)
    result_list = list(streaming_token_classification_result)
    # Convert to list to more easily check outputs
    first_result = result_list[0].results[0]
    assert isinstance(first_result, TokenClassification)
    assert first_result.start == 16
    assert first_result.end == 19
    assert first_result.word == "fox"
    assert first_result.entity == "animal"
    assert first_result.score == 0.8

    # Check processed indices
    assert result_list[0].processed_index == 19  # token - fox
    assert result_list[1].processed_index == 43  # token - dog
    assert result_list[2].processed_index == 44  # end of first sentence
    assert result_list[3].processed_index == 71  # token - land
    assert result_list[4].processed_index == len(stream_input)  # end of second sentence

    # We expect 5 results here since there are 3 tokens found
    # and the rest of each of the 2 sentences
    # (to indicate the rest of the sentences are processed)
    expected_results = 5
    count = len(result_list)
    assert count == expected_results


def test_run_bidi_stream_with_token_classification_no_results():
    """Check if model prediction with token classification
    with no results works as expected for bi-directional stream"""
    stream_input = data_model.DataStream.from_iterable(DOCUMENT)
    model = FilteredSpanClassification.bootstrap(
        lang="en",
        tokenizer=SENTENCE_TOKENIZER,
        classifier=EMPTY_RES_TOKEN_CLASSIFICATION_MODULE,
        default_threshold=0.5,
    )
    streaming_token_classification_result = model.run_bidi_stream(stream_input)
    expected_results = 2  # Sentence tokenizer returns 2 results
    count = 0
    for result in streaming_token_classification_result:
        count += 1
        # Both sentences should not have results
        assert len(result.results) == 0

    # Assert total number of results should be equal to expected number of sentences
    assert count == expected_results


def test_run_bidi_stream_chunk_stream_input():
    """Check if model prediction with token classification
    with chunks of text input works as expected for bi-directional stream"""

    chunked_document_input = (
        "The quick brown fox jumps over the ",
        "lazy dog. Once upon a time in a land far away",
    )
    stream_input = data_model.DataStream.from_iterable(chunked_document_input)
    model = FilteredSpanClassification.bootstrap(
        lang="en",
        tokenizer=SENTENCE_TOKENIZER,
        classifier=STREAM_TOKEN_CLASSIFICATION_MODULE,
        default_threshold=0.3,
    )
    streaming_token_classification_result = model.run_bidi_stream(stream_input)
    result_list = list(streaming_token_classification_result)
    # Convert to list to more easily check outputs
    first_result = result_list[0].results[0]
    assert isinstance(first_result, TokenClassification)
    assert first_result.start == 16
    assert first_result.end == 19
    assert first_result.word == "fox"
    assert first_result.entity == "animal"
    assert first_result.score == 0.8

    # Check processed indices
    assert result_list[0].processed_index == 19  # token - fox
    assert result_list[1].processed_index == 43  # token - dog
    assert result_list[2].processed_index == 44  # end of first sentence
    assert result_list[3].processed_index == 71  # token - land
    assert result_list[4].processed_index == 80  # end of second sentence

    # We expect 5 results here since there are 3 tokens found
    # and the rest of each of the 2 sentences
    # (to indicate the rest of the sentences are processed)
    expected_results = 5
    count = len(result_list)
    assert count == expected_results


def test_run_bidi_stream_with_multiple_spans_in_chunk():
    """Check if model prediction on stream with multiple sentences/spans
    works as expected for bi-directional stream"""
    doc_stream = (DOCUMENT, " I am another sentence.")
    stream_input = data_model.DataStream.from_iterable(doc_stream)
    model = FilteredSpanClassification.bootstrap(
        lang="en",
        tokenizer=SENTENCE_TOKENIZER,
        classifier=BOOTSTRAPPED_SEQ_CLASS_MODEL,
        default_threshold=0.5,
    )

    streaming_token_classification_result = model.run_bidi_stream(stream_input)
    assert isinstance(streaming_token_classification_result, Iterable)
    # Convert to list to more easily check outputs
    result_list = list(streaming_token_classification_result)

    first_result = result_list[0].results[0]
    assert isinstance(first_result, TokenClassification)
    assert first_result.start == 0
    assert first_result.end == 44
    assert first_result.word == "The quick brown fox jumps over the lazy dog."
    assert first_result.entity == "LABEL_1"
    assert approx(first_result.score) == 0.50473803
    assert result_list[1].results[0].word == "Once upon a time in a land far away."
    assert result_list[2].results[0].word == "I am another sentence."

    # Check processed indices
    assert result_list[0].processed_index == 44  # end of first sentence (in DOC)
    assert result_list[1].processed_index == 81  # end of second sentence (in DOC)

    assert result_list[2].processed_index == 104  # end of third sentence (separate)

    # Assert total number of results should be equal to expected number of sentences
    expected_number_of_sentences = 3
    count = len(result_list)
    assert count == expected_number_of_sentences


def test_run_stream_vs_no_stream():
    """Check if model prediction on stream with multiple sentences/spans
    works as expected for bi-directional stream and gives expected span results
    as non-stream"""
    multiple_sentences = (
        "The dragon hoarded gold. The cow ate grass. What is happening? What a day!"
    )
    model = FilteredSpanClassification.bootstrap(
        lang="en",
        tokenizer=SENTENCE_TOKENIZER,
        classifier=BOOTSTRAPPED_SEQ_CLASS_MODEL,
        default_threshold=0.5,
    )

    # Non-stream run
    nonstream_classification_result = model.run(multiple_sentences)
    assert len(nonstream_classification_result.results) == 4
    assert nonstream_classification_result.results[0].word == "The dragon hoarded gold."
    assert nonstream_classification_result.results[0].start == 0
    assert nonstream_classification_result.results[0].end == 24
    assert nonstream_classification_result.results[3].word == "What a day!"
    assert nonstream_classification_result.results[3].start == 63
    assert nonstream_classification_result.results[3].end == 74

    # Char-based stream
    stream_input = data_model.DataStream.from_iterable(multiple_sentences)
    stream_classification_result = model.run_bidi_stream(stream_input)
    # Convert to list to more easily check outputs
    result_list = list(stream_classification_result)
    assert len(result_list) == 4  # one per sentence
    assert result_list[0].processed_index == 24
    assert result_list[1].processed_index == 43
    assert result_list[2].processed_index == 62
    assert result_list[3].processed_index == 74

    # Chunk-based stream
    chunk_stream_input = data_model.DataStream.from_iterable((multiple_sentences,))
    chunk_stream_classification_result = model.run_bidi_stream(chunk_stream_input)
    result_list = list(chunk_stream_classification_result)
    assert len(result_list) == 4  # one per sentence
    assert result_list[0].processed_index == 24
    assert result_list[1].processed_index == 43
    assert result_list[2].processed_index == 62
    assert result_list[3].processed_index == 74
