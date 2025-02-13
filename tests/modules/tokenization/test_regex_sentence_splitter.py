"""Tests for regex sentence splitter
"""
# Standard
from typing import Iterable
import os
import tempfile

# First Party
from caikit.core import data_model
from caikit.interfaces.nlp.data_model import (
    Token,
    TokenizationResults,
    TokenizationStreamResult,
)

# Local
from caikit_nlp.modules.tokenization.regex_sentence_splitter import (
    RegexSentenceSplitter,
)

## Setup ########################################################################

# Regex sentence splitter model for reusability across tests
# NOTE: Regex may not be extremely accurate for sentence splitting needs.
REGEX_STR = "[^.!?\s][^.!?\n]*(?:[.!?](?!['\"]?\s|$)[^.!?]*)*[.!?]?['\"]?(?=\s|$)"
SENTENCE_TOKENIZER = RegexSentenceSplitter.bootstrap(REGEX_STR)
DOCUMENT = "What he told me before, I have it in my heart. I am tired of fighting."

## Tests ########################################################################


def test_bootstrap_and_run():
    """Check if we can bootstrap and run regex sentence splitter"""
    tokenization_result = SENTENCE_TOKENIZER.run(DOCUMENT)
    assert isinstance(tokenization_result, TokenizationResults)
    assert len(tokenization_result.results) == 2


def test_save_load_and_run_model():
    """Check if we can run a saved model successfully"""
    with tempfile.TemporaryDirectory() as model_dir:
        SENTENCE_TOKENIZER.save(model_dir)
        assert os.path.exists(os.path.join(model_dir, "config.yml"))

        new_splitter = RegexSentenceSplitter.load(model_dir)
        tokenization_result = new_splitter.run(DOCUMENT)
        assert isinstance(tokenization_result, TokenizationResults)
        assert len(tokenization_result.results) == 2


### Streaming tests ##############################################################


def test_run_bidi_stream_model():
    """Check if model prediction works as expected for bi-directional stream"""

    stream_input = data_model.DataStream.from_iterable(DOCUMENT)
    streaming_tokenization_result = SENTENCE_TOKENIZER.run_bidi_stream(stream_input)
    assert isinstance(streaming_tokenization_result, Iterable)
    # Convert to list to more easily check outputs
    result_list = list(streaming_tokenization_result)

    first_result = result_list[0].results[0]
    assert isinstance(first_result, Token)
    assert first_result.start == 0
    assert first_result.end == 46
    assert first_result.text == "What he told me before, I have it in my heart."

    # Check processed indices
    assert result_list[0].processed_index == 46
    assert result_list[1].processed_index == len(stream_input)

    # Assert total number of results should be equal to expected number of sentences
    expected_number_of_sentences = 2  # Sentence tokenizer returns 2 results
    count = len(result_list)
    assert count == expected_number_of_sentences


def test_run_bidi_stream_chunk_stream_input():
    """Check if model prediction with tokenization
    with chunks of text input works as expected for bi-directional stream"""

    chunked_document_input = (
        "What he told me ",
        "before, I have it in my heart. I am tired of fighting. ",
        " The cow jumped over the moon. ",
    )
    stream_input = data_model.DataStream.from_iterable(chunked_document_input)
    streaming_tokenization_result = SENTENCE_TOKENIZER.run_bidi_stream(stream_input)
    result_list = list(streaming_tokenization_result)
    # Convert to list to more easily check outputs
    first_result = result_list[0].results[0]
    assert isinstance(first_result, Token)
    assert first_result.start == 0
    assert first_result.end == 46
    assert first_result.text == "What he told me before, I have it in my heart."

    # Check processed indices
    assert result_list[0].processed_index == 46  # ...heart.
    assert result_list[1].processed_index == 71  # ...fighting.
    assert result_list[2].processed_index == 102  # end of doc

    expected_results = 3
    count = len(result_list)
    assert count == expected_results


def test_run_bidi_stream_empty():
    """Check if tokenization can run with empty space for streaming"""
    stream_input = data_model.DataStream.from_iterable("")
    streaming_tokenization_result = SENTENCE_TOKENIZER.run_bidi_stream(stream_input)
    assert isinstance(streaming_tokenization_result, Iterable)
    # Convert to list to more easily check outputs
    result_list = list(streaming_tokenization_result)
    assert len(result_list) == 1
    assert result_list[0].results == []
    assert result_list[0].processed_index == 0
