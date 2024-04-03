"""Tests for text embedding module"""

# Standard
from typing import List, Tuple
import os
import tempfile

# Third Party
from pytest import approx
from torch.backends import mps
import numpy as np
import pytest
import torch

# First Party
from caikit.core import ModuleConfig
from caikit.interfaces.common.data_model.vectors import ListOfVector1D
from caikit.interfaces.nlp.data_model import (
    EmbeddingResult,
    RerankResult,
    RerankResults,
    RerankScore,
    RerankScores,
)

# Local
from caikit_nlp.modules.text_embedding import EmbeddingModule, utils
from caikit_nlp.modules.text_embedding.embedding import (
    get_sample_start_indexes,
    sum_token_count,
)
from tests.fixtures import SEQ_CLASS_MODEL

## Setup ########################################################################

# Bootstrapped sequence classification model for reuse across tests
# .bootstrap is tested separately in the first test
BOOTSTRAPPED_MODEL = EmbeddingModule.bootstrap(SEQ_CLASS_MODEL)

# Token counts:
# All expected token counts were calculated with reference to the
# `BertForSequenceClassification` model. Each model's tokenizer behaves differently
# which can lead to the expected token counts being invalid.

INPUT = "The quick brown fox jumps over the lazy dog."
INPUT_TOKEN_COUNT = 36 + 2  # [CLS] Thequickbrownfoxjumpsoverthelazydog. [SEP]

MANY_INPUTS = [
    "The quick brown fox jumps over the lazy dog.",
    "But I must explain to you how all this mistaken idea.",
    "No one rejects or dislikes.",
]

QUERY = "What is foo bar?"
QUERY_TOKEN_COUNT = 13 + 2  # [CLS] Whatisfoobar? [SEP]

QUERIES: List[str] = [
    "Who is foo?",
    "Where is the bar?",
]
QUERIES_TOKEN_COUNT = (9 + 2) + (
    14 + 2
)  # [CLS] Whoisfoo? [SEP], [CLS] Whereisthebar? [SEP]


# These are used to test that documents can handle different types in and out
TYPE_KEYS = "str_test", "int_test", "float_test", "nested_dict_test"

DOCS = [
    {
        "text": "foo",
        "title": "title or whatever",
        "str_test": "test string",
        "int_test": 1,
        "float_test": 1.234,
        "score": 99999,
        "nested_dict_test": {"deep1": 1, "deep string": "just testing"},
    },
    {
        "_text": "bar",
        "title": "title 2",
    },
    {
        "text": "foo and bar",
    },
    {
        "_text": "Where is the bar",
        "another": "something else",
    },
]

# The `text` and `_text` keys are extracted from DOCS as input to the tokenizer
# [CLS] foo [SEP], [CLS] bar [SEP], [CLS] fooandbar [SEP], [CLS] Whereisthebar [SEP]
DOCS_TOKEN_COUNT = (3 + 2) + (3 + 2) + (9 + 2) + (13 + 2)

# Use text or _text from DOCS for our test sentences
SENTENCES = [d.get("text", d.get("_text")) for d in DOCS]

# [CLS] foo [SEP], [CLS] bar [SEP], [CLS] fooandbar [SEP], [CLS] Whereisthebar [SEP]
SENTENCES_TOKEN_COUNT = (3 + 2) + (3 + 2) + (9 + 2) + (13 + 2)

## Tests ########################################################################


@pytest.fixture(scope="module", name="loaded_model")
def fixture_loaded_model(tmp_path_factory):
    models_dir = tmp_path_factory.mktemp("models")
    model_path = str(models_dir / "model_id")
    BOOTSTRAPPED_MODEL.save(model_path)
    model = EmbeddingModule.load(model_path)
    return model


def _assert_is_expected_vector(vector):
    assert isinstance(vector.data.values[0], np.float32)
    assert len(vector.data.values) == 32
    # Just testing a few values for readability
    assert approx(vector.data.values[0]) == 0.3244932293891907
    assert approx(vector.data.values[1]) == -0.4934631288051605
    assert approx(vector.data.values[2]) == 0.5721234083175659


def _assert_is_expected_embedding_result(actual):
    assert isinstance(actual, EmbeddingResult)
    vector = actual.result
    _assert_is_expected_vector(vector)


def _assert_is_expected_embeddings_results(actual):
    assert isinstance(actual, ListOfVector1D)
    _assert_is_expected_vector(actual.vectors[0])


def test_bootstrap():
    assert isinstance(
        EmbeddingModule.bootstrap(SEQ_CLASS_MODEL), EmbeddingModule
    ), "bootstrap error"


def _assert_types_found(types_found):
    assert type(types_found["str_test"]) == str, "passthru str value type check"
    assert type(types_found["int_test"]) == int, "passthru int value type check"
    assert type(types_found["float_test"]) == float, "passthru float value type check"
    assert (
        type(types_found["nested_dict_test"]) == dict
    ), "passthru nested dict value type check"


def _assert_valid_scores(scores, type_tests={}):
    for score in scores:
        assert isinstance(score, RerankScore)
        assert isinstance(score.score, float)
        assert isinstance(score.index, int)
        assert isinstance(score.text, str)

        document = score.document
        assert isinstance(document, dict)
        assert document == DOCS[score.index]

        # Test document key named score (None or 9999) is independent of the result score
        assert score.score != document.get(
            "score"
        ), "unexpected passthru score same as result score"

        # Gather various type test values when we have them
        for k, v in document.items():
            if k in TYPE_KEYS:
                type_tests[k] = v

    return type_tests


def test_bootstrap_model(loaded_model):
    assert isinstance(BOOTSTRAPPED_MODEL, EmbeddingModule), "bootstrap model type"
    assert (
        BOOTSTRAPPED_MODEL.model.__class__.__name__ == "SentenceTransformer"
    ), "bootstrap model class name"
    # worth noting that bootstrap does not wrap, but load does
    assert (
        loaded_model.model.__class__.__name__ == "SentenceTransformerWithTruncate"
    ), "loaded model class name"


def test_save_load_and_run():
    """Check if we can load and run a saved model successfully"""
    model_id = "model_id"
    with tempfile.TemporaryDirectory(suffix="-1st") as model_dir:
        model_path = os.path.join(model_dir, model_id)
        BOOTSTRAPPED_MODEL.save(model_path)
        new_model = EmbeddingModule.load(model_path)

    assert isinstance(new_model, EmbeddingModule), "save and load error"
    assert new_model != BOOTSTRAPPED_MODEL, "did not load a new model"

    # Use run_embedding just to make sure this new model is usable
    result = new_model.run_embedding(text=INPUT)
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


def test_load_without_artifacts():
    """Test coverage for the error message when config has no artifacts to load"""
    with pytest.raises(ValueError):
        EmbeddingModule.load(ModuleConfig({}))


def test_run_embedding_type_check(loaded_model):
    """Input cannot be a list"""
    with pytest.raises(TypeError):
        loaded_model.run_embedding([INPUT])
        pytest.fail("Should not reach here")


def test_run_embedding(loaded_model):
    res = loaded_model.run_embedding(text=INPUT)
    _assert_is_expected_embedding_result(res)
    assert res.input_token_count == INPUT_TOKEN_COUNT


def test_run_embeddings_str_type(loaded_model):
    """Supposed to be a list, gets fixed automatically."""
    res = loaded_model.run_embeddings(texts=INPUT)
    assert isinstance(res.results.vectors, list)
    assert len(res.results.vectors) == 1


def test_run_embeddings(loaded_model):
    res = loaded_model.run_embeddings(texts=[INPUT])
    assert isinstance(res.results.vectors, list)
    _assert_is_expected_embeddings_results(res.results)
    assert res.input_token_count == INPUT_TOKEN_COUNT


@pytest.mark.parametrize(
    "query,docs,top_n",
    [
        (["test list"], DOCS, None),
        (None, DOCS, 1234),
        (False, DOCS, 1234),
        (QUERY, {"testdict": "not list"}, 1234),
        (QUERY, DOCS, "topN string is not an integer or None"),
    ],
)
def test_run_rerank_query_type_error(query, docs, top_n, loaded_model):
    """test for type checks matching task/run signature"""
    with pytest.raises(TypeError):
        loaded_model.run_rerank_query(query=query, documents=docs, top_n=top_n)
        pytest.fail("Should not reach here.")


def test_run_rerank_query_no_type_error(loaded_model):
    """no type error with list of string queries and list of dict documents"""
    res = loaded_model.run_rerank_query(query=QUERY, documents=DOCS, top_n=1)
    assert res.input_token_count == QUERY_TOKEN_COUNT + DOCS_TOKEN_COUNT


@pytest.mark.parametrize(
    "top_n, expected",
    [
        (1, 1),
        (2, 2),
        (None, len(DOCS)),
        (-1, len(DOCS)),
        (0, len(DOCS)),
        (9999, len(DOCS)),
    ],
)
def test_run_rerank_query_top_n(top_n, expected, loaded_model):
    res = loaded_model.run_rerank_query(query=QUERY, documents=DOCS, top_n=top_n)
    assert isinstance(res, RerankResult)
    assert len(res.result.scores) == expected
    assert res.input_token_count == QUERY_TOKEN_COUNT + DOCS_TOKEN_COUNT


def test_run_rerank_query_no_query(loaded_model):
    with pytest.raises(TypeError):
        loaded_model.run_rerank_query(query=None, documents=DOCS, top_n=99)


def test_run_rerank_query_zero_docs(loaded_model):
    """No empty doc list therefore result is zero result scores"""
    with pytest.raises(ValueError):
        loaded_model.run_rerank_query(query=QUERY, documents=[], top_n=99)


def test_run_rerank_query(loaded_model):
    res = loaded_model.run_rerank_query(query=QUERY, documents=DOCS)
    assert isinstance(res, RerankResult)

    scores = res.result.scores
    assert isinstance(scores, list)
    assert len(scores) == len(DOCS)

    types_found = _assert_valid_scores(scores)
    _assert_types_found(types_found)
    assert res.input_token_count == QUERY_TOKEN_COUNT + DOCS_TOKEN_COUNT


@pytest.mark.parametrize(
    "queries,docs", [("test string", DOCS), (QUERIES, {"testdict": "not list"})]
)
def test_run_rerank_queries_type_error(queries, docs, loaded_model):
    """type error check ensures params are lists and not just 1 string or just one doc (for example)"""
    with pytest.raises(TypeError):
        loaded_model.run_rerank_queries(queries=queries, documents=docs)
        pytest.fail("Should not reach here.")


def test_run_rerank_queries_no_type_error(loaded_model):
    """no type error with list of string queries and list of dict documents"""
    res = loaded_model.run_rerank_queries(queries=QUERIES, documents=DOCS, top_n=99)
    assert res.input_token_count == QUERIES_TOKEN_COUNT + DOCS_TOKEN_COUNT


@pytest.mark.parametrize(
    "top_n, expected",
    [
        (1, 1),
        (2, 2),
        (None, len(DOCS)),
        (-1, len(DOCS)),
        (0, len(DOCS)),
        (9999, len(DOCS)),
    ],
)
def test_run_rerank_queries_top_n(top_n, expected, loaded_model):
    """no type error with list of string queries and list of dict documents"""
    res = loaded_model.run_rerank_queries(queries=QUERIES, documents=DOCS, top_n=top_n)
    assert isinstance(res, RerankResults)
    assert len(res.results) == len(QUERIES)
    for result in res.results:
        assert len(result.scores) == expected
    assert res.input_token_count == QUERIES_TOKEN_COUNT + DOCS_TOKEN_COUNT


@pytest.mark.parametrize(
    "queries, docs",
    [
        ([], DOCS),
        (QUERIES, []),
        ([], []),
    ],
    ids=["no queries", "no docs", "no queries and no docs"],
)
def test_run_rerank_queries_no_queries_or_no_docs(queries, docs, loaded_model):
    """No queries and/or no docs therefore result is zero results"""

    with pytest.raises(ValueError):
        loaded_model.run_rerank_queries(queries=queries, documents=docs, top_n=9)


def test_run_rerank_queries(loaded_model):
    top_n = 2
    rerank_result = loaded_model.run_rerank_queries(
        queries=QUERIES, documents=DOCS, top_n=top_n
    )
    assert isinstance(rerank_result, RerankResults)

    results = rerank_result.results
    assert isinstance(results, list)
    assert len(results) == 2 == len(QUERIES)  # 2 queries yields 2 result(s)

    types_found = {}  # Gather the type tests from any of the results

    for result in results:
        assert isinstance(result, RerankScores)
        scores = result.scores
        assert isinstance(scores, list)
        assert len(scores) == top_n
        types_found = _assert_valid_scores(scores, types_found)

    # Make sure our document fields of different types made it in/out ok
    _assert_types_found(types_found)
    assert rerank_result.input_token_count == QUERIES_TOKEN_COUNT + DOCS_TOKEN_COUNT


def test_run_sentence_similarity(loaded_model):
    res = loaded_model.run_sentence_similarity(
        source_sentence=QUERY, sentences=SENTENCES
    )
    scores = res.result.scores
    assert len(scores) == len(SENTENCES)
    for score in scores:
        assert isinstance(score, float)
    assert res.input_token_count == QUERY_TOKEN_COUNT + SENTENCES_TOKEN_COUNT


def test_run_sentence_similarities(loaded_model):
    res = loaded_model.run_sentence_similarities(
        source_sentences=QUERIES, sentences=SENTENCES
    )
    results = res.results
    assert len(results) == len(QUERIES)
    for result in results:
        scores = result.scores
        assert len(scores) == len(SENTENCES)
        for score in scores:
            assert isinstance(score, float)
    assert res.input_token_count == QUERIES_TOKEN_COUNT + SENTENCES_TOKEN_COUNT


@pytest.mark.parametrize(
    "use_ipex, device, expected",
    [
        (True, "", None),
        (False, "", None),
        (True, None, None),
        (False, None, None),
        (False, "xpu", None),
        (True, "xpu", "xpu"),
        (True, "mps", None),
        (
            False,
            "mps",
            "mps" if mps.is_built() and mps.is_available() else None,
        ),
    ],
)
def test__select_device(use_ipex, device, expected):
    assert EmbeddingModule._select_device(use_ipex, device) == expected


@pytest.mark.parametrize(
    "use_ipex, use_device, expected",
    [
        (True, None, "ipex"),
        (True, "mps", "ipex"),
        (False, "mps", mps),
        (False, None, "inductor"),
    ],
)
def test__get_backend(use_ipex, use_device, expected):
    # Make the Mac MPS test work depending on availability
    assert EmbeddingModule._get_backend(use_ipex, use_device) == expected


@pytest.mark.parametrize(
    "use_ipex",
    [None, "true", "True", "False", "false"],
)
def test__get_ipex(use_ipex):
    """Test that _get_ipex returns False instead of raising an exception.

    Assumes that when running tests, we won't have IPEX installed.
    """
    assert not EmbeddingModule._get_ipex(use_ipex)


def test__optimize():
    """Test that _optimize does nothing when disabled"""
    fake = "fake model"  # Will be returned as-is
    assert fake == EmbeddingModule._optimize(fake, False, "bogus", False, False)


@pytest.mark.parametrize("truncate_input_tokens", [0, 513])
def test__truncate_input_tokens_raises(truncate_input_tokens, loaded_model):
    model_max = loaded_model.model.max_seq_length

    too_long = "x " * (model_max - 1)  # This will go over
    over = model_max + 1
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.model.encode(
            sentences=[too_long], truncate_input_tokens=truncate_input_tokens
        )
    # Same behavior when implicit_truncation_errors is True (the default)
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.model.encode(
            sentences=[too_long],
            truncate_input_tokens=truncate_input_tokens,
            implicit_truncation_errors=True,
        )
    # Different behavior when implicit_truncation_errors is False -- no error raised!
    loaded_model.model.encode(
        sentences=[too_long],
        truncate_input_tokens=truncate_input_tokens,
        implicit_truncation_errors=False,
    )


def test__implicit_truncation(loaded_model):
    """Test that implicit truncation happens (when allowed)"""
    model_max = loaded_model.model.max_seq_length

    too_long = "x " * (model_max - 1)  # This will go over a little
    extra_long = (
        too_long
        + "more clever words that surely change the meaning of this text"
        * (model_max - 1)
    )

    # Allowed truncation using default tokens (0) and config to disable the error.
    res = loaded_model.model.encode(
        sentences=[too_long], truncate_input_tokens=0, implicit_truncation_errors=False
    )
    # Allowed truncation using model max
    res_extra_max = loaded_model.model.encode(
        sentences=[extra_long], truncate_input_tokens=loaded_model.model.max_seq_length
    )
    # Allowed truncation using -1 to just let the model do its thing
    res_extra_neg = loaded_model.model.encode(
        sentences=[extra_long], truncate_input_tokens=-1
    )

    # Demonstrating that when implicit truncation is allowed, sentence-transformers is quietly truncating at model max
    # The simple too_long string of x's, is equivalent to the string with significantly different extra text (truncated)
    assert np.allclose(res, res_extra_max)
    assert np.allclose(res, res_extra_neg)


def test_not_too_many_tokens(loaded_model):
    """Happy path for the endpoints using text that is not too many tokens."""

    model_max = loaded_model.model.max_seq_length

    ok = "x " * (model_max - 2)  # Subtract 2 for begin/end tokens

    # embedding(s)
    loaded_model.run_embedding(text=ok)
    loaded_model.run_embeddings(texts=[ok])

    # sentence similarity(ies) test both source_sentence and sentences
    loaded_model.run_sentence_similarity(source_sentence=ok, sentences=[ok])
    loaded_model.run_sentence_similarities(source_sentences=[ok], sentences=[ok])

    # reranker test both query and document text
    loaded_model.run_rerank_query(query=ok, documents=[{"text": ok}])
    loaded_model.run_rerank_queries(queries=[ok], documents=[{"text": ok}])


def test_too_many_tokens_default(loaded_model):
    """These endpoints raise an error when truncation would happen."""

    model_max = loaded_model.model.max_seq_length
    over = model_max + 1

    ok = "x " * (model_max - 2)  # Subtract 2 for begin/end tokens
    too_long = "x " * (model_max - 1)  # This will go over

    # embedding(s)
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_embedding(text=too_long)
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_embeddings(texts=[too_long])

    # sentence similarity(ies) test both source_sentence and sentences
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_sentence_similarity(source_sentence=too_long, sentences=[ok])
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_sentence_similarity(source_sentence=ok, sentences=[too_long])

    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_sentence_similarities(
            source_sentences=[too_long], sentences=[ok]
        )
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_sentence_similarities(
            source_sentences=[ok], sentences=[too_long]
        )

    # reranker test both query and document text
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_rerank_query(query=too_long, documents=[{"text": ok}])
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_rerank_query(query=ok, documents=[{"text": too_long}])

    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_rerank_queries(queries=[too_long], documents=[{"text": ok}])
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_rerank_queries(queries=[ok], documents=[{"text": too_long}])


@pytest.mark.parametrize("truncate_input_tokens", [0, 513])
def test_too_many_tokens_error_params(truncate_input_tokens, loaded_model):
    """truncate_input_tokens does not prevent these endpoints from raising an error.

    Test with 0 which uses the max model len (512) to determine truncation and raise error.
    Test with 513 (> 512) which detects truncation over 512 and raises an error.
    """

    model_max = loaded_model.model.max_seq_length
    over = model_max + 1

    ok = "x " * (model_max - 2)  # Subtract 2 for begin/end tokens
    too_long = "x " * (model_max - 1)  # This will go over

    # embedding(s)
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_embedding(
            text=too_long, truncate_input_tokens=truncate_input_tokens
        )
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_embeddings(
            texts=[too_long], truncate_input_tokens=truncate_input_tokens
        )

    # sentence similarity(ies) test both source_sentence and sentences
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_sentence_similarity(
            source_sentence=too_long,
            sentences=[ok],
            truncate_input_tokens=truncate_input_tokens,
        )
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_sentence_similarity(
            source_sentence=ok,
            sentences=[too_long],
            truncate_input_tokens=truncate_input_tokens,
        )

    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_sentence_similarities(
            source_sentences=[too_long],
            sentences=[ok],
            truncate_input_tokens=truncate_input_tokens,
        )
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_sentence_similarities(
            source_sentences=[ok],
            sentences=[too_long],
            truncate_input_tokens=truncate_input_tokens,
        )

    # reranker test both query and document text
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_rerank_query(
            query=too_long,
            documents=[{"text": ok}],
            truncate_input_tokens=truncate_input_tokens,
        )
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_rerank_query(
            query=ok,
            documents=[{"text": too_long}],
            truncate_input_tokens=truncate_input_tokens,
        )

    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_rerank_queries(
            queries=[too_long],
            documents=[{"text": ok}],
            truncate_input_tokens=truncate_input_tokens,
        )
    with pytest.raises(ValueError, match=f"({over} > {model_max})"):
        loaded_model.run_rerank_queries(
            queries=[ok],
            documents=[{"text": too_long}],
            truncate_input_tokens=truncate_input_tokens,
        )


@pytest.mark.parametrize("truncate_input_tokens", [-1, 99, 510, 511, 512])
def test_too_many_tokens_with_truncation_working(truncate_input_tokens, loaded_model):
    """truncate_input_tokens prevents these endpoints from raising an error when too many tokens.

    Test with -1 which lets the model do truncation instead of raising an error.
    Test with 99 (< 512 -2) which causes our code to do the truncation instead of raising an error.
    Test with 510 (512 -2) which causes our code to do the truncation instead of raising an error.
    511 and 512 also behave like 510. The value is allowed, but begin/end tokens will take space.
    """

    model_max = loaded_model.model.max_seq_length

    ok = "x " * (model_max - 2)  # Subtract 2 for begin/end tokens
    too_long = "x " * (model_max - 1)  # This will go over

    # embedding(s)
    loaded_model.run_embedding(
        text=too_long, truncate_input_tokens=truncate_input_tokens
    )
    loaded_model.run_embeddings(
        texts=[too_long], truncate_input_tokens=truncate_input_tokens
    )

    # sentence similarity(ies) test both source_sentence and sentences
    loaded_model.run_sentence_similarity(
        source_sentence=too_long,
        sentences=[ok],
        truncate_input_tokens=truncate_input_tokens,
    )
    loaded_model.run_sentence_similarity(
        source_sentence=ok,
        sentences=[too_long],
        truncate_input_tokens=truncate_input_tokens,
    )

    loaded_model.run_sentence_similarities(
        source_sentences=[too_long],
        sentences=[ok],
        truncate_input_tokens=truncate_input_tokens,
    )
    loaded_model.run_sentence_similarities(
        source_sentences=[ok],
        sentences=[too_long],
        truncate_input_tokens=truncate_input_tokens,
    )

    # reranker test both query and document text
    loaded_model.run_rerank_query(
        query=too_long,
        documents=[{"text": ok}],
        truncate_input_tokens=truncate_input_tokens,
    )
    loaded_model.run_rerank_query(
        query=ok,
        documents=[{"text": too_long}],
        truncate_input_tokens=truncate_input_tokens,
    )

    loaded_model.run_rerank_queries(
        queries=[too_long],
        documents=[{"text": ok}],
        truncate_input_tokens=truncate_input_tokens,
    )
    loaded_model.run_rerank_queries(
        queries=[ok],
        documents=[{"text": too_long}],
        truncate_input_tokens=truncate_input_tokens,
    )


@pytest.mark.parametrize(
    "truncate_input_tokens", [1, 2, 3, 4, 99, 100, 101, 510, 511, 512, -1]
)
def test_embeddings_with_truncation(truncate_input_tokens, loaded_model):
    """verify that results are as expected with truncation"""

    max_len = loaded_model.model.max_seq_length - 2
    if truncate_input_tokens is None or truncate_input_tokens < 0:
        # For -1 we don't truncate, but sentence-transformers will truncate at max_seq_length - 2
        repeat = max_len
    else:
        repeat = min(
            truncate_input_tokens, max_len
        )  # max_len is used when we need -2 for begin/end

    # Build a text like "x x x.. x " with room for one more token
    repeat = repeat - 1  # space for the final x or y token to show difference

    base = ""
    if repeat > 0:
        base = "x " * repeat  # A bunch of "x" tokens
    x = base + "x"  # One last "x" that will not get truncated
    y = base + "y"  # A different last character "y" not truncated
    z = y + "z"  # Add token "z" after "y". This should get truncated.

    res = loaded_model.run_embeddings(
        texts=[base, x, y, z], truncate_input_tokens=truncate_input_tokens
    )
    vectors = res.results.vectors  # vectors from batch embeddings

    # Compare with results from individual embedding calls in a loop
    loop_res = []
    for t in [base, x, y, z]:
        r = loaded_model.run_embedding(
            text=t, truncate_input_tokens=truncate_input_tokens
        )
        loop_res.append(r)
    loop_vectors = [
        r.result for r in loop_res
    ]  # vectors from loop of single embedding calls

    assert len(vectors) == len(loop_vectors), "expected the same length vectors"
    # compare the vectors from batch with the single calls
    for i, e in enumerate(vectors):
        assert np.allclose(e.data.values, loop_vectors[i].data.values)

    # x...xyz is the same as x...xy because that is exactly where truncation worked
    assert len(vectors[2].data.values) == len(vectors[3].data.values)
    assert np.allclose(vectors[2].data.values, vectors[3].data.values)
    for i in range(len(vectors[2].data.values)):
        assert approx(vectors[2].data.values[i]) == approx(vectors[3].data.values[i])

    # Make sure the base, x, y are not a match (we kept the significant last char)
    assert not np.allclose(vectors[0].data.values, vectors[1].data.values)
    assert not np.allclose(vectors[0].data.values, vectors[2].data.values)
    assert not np.allclose(vectors[1].data.values, vectors[2].data.values)


def test__with_retry_happy_path(loaded_model):
    """works with args/kwargs, no problems"""
    loaded_model._with_retry(print, "hello", "world", sep="<:)>", end="!!!\n")


def test__with_retry_fail(loaded_model):
    """fn never works, loops then raises the exception"""

    def fn():
        raise (ValueError("always fails with ValueError"))

    with pytest.raises(ValueError):
        loaded_model._with_retry(fn)


def test__with_retry_fail_fail(loaded_model, monkeypatch):
    """fn needs a few tries, tries twice and fails."""

    monkeypatch.setattr(loaded_model, "RETRY_COUNT", 1)  # less than 3 tries

    def generate_ints():
        yield from range(9)  # More than enough for retry loop

    ints = generate_ints()

    def fail_fail_win():
        for i in ints:
            if i < 2:  # fail, fail
                raise (ValueError(f"fail {i}"))
            else:  # win and return 3
                return i + 1

    # Without a third try raises first exception
    with pytest.raises(ValueError) as e:
        loaded_model._with_retry(fail_fail_win)

    assert e.value.args[0] == "fail 0", "expected first exception 'fail 0'"


def test__with_retry_fail_fail_win(loaded_model, monkeypatch):
    """fn needs a few tries, logs, loops and succeeds"""

    monkeypatch.setattr(loaded_model, "RETRY_COUNT", 6)  # test needs at least 3 tries

    def generate_ints():
        yield from range(9)  # More than enough for retry loop

    ints = generate_ints()

    def fail_fail_win():
        for i in ints:
            if i < 2:  # fail, fail
                raise (ValueError("fail, fail"))
            else:  # win and return 3
                return i + 1

    # Third try did not raise an exception. Returns 3.
    assert 3 == loaded_model._with_retry(fail_fail_win)


def test_env_val_to_bool():
    assert not utils.env_val_to_bool(None)
    assert not utils.env_val_to_bool("")
    assert not utils.env_val_to_bool("   ")
    assert not utils.env_val_to_bool(0)
    assert not utils.env_val_to_bool("0")
    assert not utils.env_val_to_bool(" False ")
    assert not utils.env_val_to_bool("  false   ")
    assert not utils.env_val_to_bool("   fAlSE    ")

    assert utils.env_val_to_bool(1)
    assert utils.env_val_to_bool("1")
    assert utils.env_val_to_bool(" True ")
    assert utils.env_val_to_bool("  true   ")
    assert utils.env_val_to_bool("   tRuE    ")


def test_env_val_to_int():
    expected_default = 12345
    assert expected_default == utils.env_val_to_int(None, expected_default)
    assert expected_default == utils.env_val_to_int("", expected_default)
    assert expected_default == utils.env_val_to_int("   ", expected_default)
    assert expected_default == utils.env_val_to_int(" ss ", expected_default)
    assert expected_default == utils.env_val_to_int("  sss   ", expected_default)
    assert expected_default == utils.env_val_to_int("   ssss    ", expected_default)

    assert 0 == utils.env_val_to_int(0, expected_default)
    assert 0 == utils.env_val_to_int("0", expected_default)
    assert 0 == utils.env_val_to_int(False, expected_default)
    assert 456 == utils.env_val_to_int("456", expected_default)
    assert 456 == utils.env_val_to_int(" 456 ", expected_default)
    assert 1 == utils.env_val_to_int(True, expected_default)


@pytest.mark.parametrize(
    # `expected_count` are valid for the `BertForSequenceClassification` model.
    ["texts", "expected_count"],
    [
        # Only tokens requiring model attention is counted.
        # [PAD] doesn't attract model attention, but [CLS] and [SEP] does
        # [CLS] 5 normal tokens [SEP]
        (["12345"], 5 + 2),
        # [CLS] 5 normal [SEP], [CLS] 4 normal [SEP] [PAD]
        (["12 345", "6 789"], 9 + 4),
    ],
)
def test_sum_token_count_no_truncation(texts, expected_count, loaded_model):

    tokenized = loaded_model.model.tokenizer(
        texts,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_length=True,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=loaded_model.model.max_seq_length,
    )
    token_count = sum_token_count(
        tokenized,
        truncate_only=False,
    )

    assert token_count == expected_count


@pytest.mark.parametrize(
    # `expected_count` are valid for the `BertForSequenceClassification` model.
    ["texts", "truncate", "expected_count"],
    [
        # Only tokens requiring model attention is counted.
        # [PAD] doesn't attract model attention, but [CLS] and [SEP] does
        #
        # All encodings: [CLS] 12345 [SEP]
        # No truncation
        (["12345"], 10, 7),
        # All encodings: [CLS] 123 [SEP] + [CLS] 45 [SEP] [PAD]
        # Only truncated: [CLS] 123 [SEP]
        (["12345"], 5, 3 + 2),
        #
        # All encodings: [CLS] 123 [SEP] + [CLS] 45 [SEP] [PAD], [CLS] 678 [SEP] + [CLS] 9 [SEP] [PAD] [PAD]
        # Only truncated: [CLS] 123 [SEP] , [CLS] 678 [SEP]
        (["12 345", "6 789"], 5, (3 + 2) + (3 + 2)),
    ],
)
def test_sum_token_count_with_truncation(texts, truncate, expected_count, loaded_model):
    tokenized = loaded_model.model.tokenizer(
        texts,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_length=True,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=truncate,
    )
    token_count = sum_token_count(
        tokenized,
        truncate_only=True,
    )

    assert token_count == expected_count


@pytest.mark.parametrize(
    "truncate_input_tokens", [0, 1, 2, 3, 4, 99, 100, 101, 510, 511, 512, 513, -1]
)
def test_encoding_order(loaded_model: EmbeddingModule, truncate_input_tokens):
    """Confirm that encoding doesn't modify the original sort order"""
    separate_embeddings = [
        loaded_model.run_embedding(text=i, truncate_input_tokens=truncate_input_tokens)
        for i in MANY_INPUTS
    ]
    combined_embeddings = loaded_model.run_embeddings(
        texts=MANY_INPUTS, truncate_input_tokens=truncate_input_tokens
    )

    separate_vectors = [
        e.to_dict()["result"]["data"]["values"] for e in separate_embeddings
    ]
    combined_vectors = [
        e["data"]["values"] for e in combined_embeddings.to_dict()["results"]["vectors"]
    ]

    assert len(separate_vectors) == len(
        combined_vectors
    ), "expected the same number separate and combined embeddings"

    # test order by comparing value of individual embeddings in sequence
    for i, e in enumerate(separate_vectors):
        assert np.allclose(e, combined_vectors[i])

    # test expected failure case by reordering
    shifted_separate_vectors = separate_vectors[1:] + [separate_vectors[0]]

    for i, e in enumerate(shifted_separate_vectors):
        assert e != separate_vectors[i], "expected order to be have been altered"
        assert (
            not approx(e) == combined_vectors[i]
        ), "expected altered order to not match combined vectors"
        assert not np.allclose(
            e, combined_vectors[i]
        ), "expected altered order to not match combined"


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        ([0, 0, 0, 0, 0], [0]),
        ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
        ([0, 0, 1, 1, 1, 2], [0, 2, 5]),
        ([], []),
    ],
)
def test_get_sample_start_indexes(mapping, expected):
    mock_tokenized = {
        "overflow_to_sample_mapping": torch.Tensor(mapping).type(torch.int8)
    }
    assert get_sample_start_indexes(mock_tokenized) == expected


def test_encode_extensions(loaded_model):
    # loaded model can return_token_count
    ret = loaded_model._encode_with_retry("text here", return_token_count=True)
    assert isinstance(ret, Tuple)
    assert isinstance(ret[0], np.ndarray)
    assert isinstance(ret[1], int)
    ret = loaded_model._encode_with_retry("text here", return_token_count=False)
    assert isinstance(ret, np.ndarray)

    # Make sure use with un-wrapped SentenceTransformer model is unaffected by extended params or return tokens
    ret = BOOTSTRAPPED_MODEL._encode_with_retry(
        "text here",
        return_token_count=True,
        truncate_input_tokens=123,
        implicit_truncation_errors=False,
    )
    assert isinstance(ret, np.ndarray)
    BOOTSTRAPPED_MODEL._encode_with_retry(
        "text here"
    )  # and no KeyError trying to remove non-existing keys


@pytest.mark.parametrize(
    "truncate_input_tokens",
    [0, 1, 2, 3, 4, 5, 99, 100, 101, 300, 510, 511, 512, 513, 1000, -1],
)
def test_same_same(loaded_model: EmbeddingModule, truncate_input_tokens):
    """Confirm that same text gives same results"""

    inputs = ["What is generative ai?", "What is generative ai?", "different"]

    # First ensuring that batch input vs loop over inputs is the same
    separate_embeddings = [
        loaded_model.run_embedding(text=i, truncate_input_tokens=truncate_input_tokens)
        for i in inputs
    ]
    combined_embeddings = loaded_model.run_embeddings(
        texts=inputs, truncate_input_tokens=truncate_input_tokens
    )

    separate_vectors = [
        e.to_dict()["result"]["data"]["values"] for e in separate_embeddings
    ]
    combined_vectors = [
        e["data"]["values"] for e in combined_embeddings.to_dict()["results"]["vectors"]
    ]

    assert len(separate_vectors) == len(
        combined_vectors
    ), "expected the same number separate and combined embeddings"

    # test order by comparing value of individual embeddings in sequence
    for i, e in enumerate(separate_vectors):
        assert np.allclose(e, combined_vectors[i])

    # Next ensuring that the two identical sentences yield identical results (and 3rd does not)
    assert np.array_equal(combined_vectors[0], combined_vectors[1])
    assert not np.array_equal(combined_vectors[1], combined_vectors[2])
    assert np.array_equal(separate_vectors[0], separate_vectors[1])
    assert not np.array_equal(separate_vectors[1], separate_vectors[2])
