"""Tests for regex sentence splitter
"""
# Standard
import os
import tempfile

# First Party
from caikit.interfaces.nlp.data_model import TokenizationResults

# Local
from caikit_nlp.modules.tokenization.regex_sentence_splitter import (
    RegexSentenceSplitter,
)

## Setup ########################################################################

# Regex sentence splitter model for reusability across tests
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
