"""Tests for text embedding module
"""
# Standard
from typing import List
import os
import tempfile

# Third Party
from pytest import approx
import numpy as np
import pytest

# First Party
from caikit.core import ModuleConfig

# Local
from caikit_nlp.data_model import (
    EmbeddingResult,
    ListOfVector1D,
    RerankPredictions,
    RerankQueryResult,
    RerankScore,
    Vector1D,
)
from caikit_nlp.modules.text_embedding import EmbeddingModule
from tests.fixtures import SEQ_CLASS_MODEL

## Setup ########################################################################

# Bootstrapped sequence classification model for reuse across tests
# .bootstrap is tested separately in the first test
BOOTSTRAPPED_MODEL = EmbeddingModule.bootstrap(SEQ_CLASS_MODEL)

INPUT = "The quick brown fox jumps over the lazy dog."

QUERY = "What is foo bar?"

QUERIES: List[str] = [
    "Who is foo?",
    "Where is the bar?",
]

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

# Use text or _text from DOCS for our test sentences
SENTENCES = [d.get("text", d.get("_text")) for d in DOCS]

## Tests ########################################################################


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
    vectors = actual.results
    _assert_is_expected_vector(vectors[0])


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


def test_bootstrap():
    assert isinstance(BOOTSTRAPPED_MODEL, EmbeddingModule), "bootstrap error"


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


def test_run_embedding_type_check():
    """Input cannot be a list"""
    model = BOOTSTRAPPED_MODEL
    with pytest.raises(TypeError):
        model.run_embedding([INPUT])
        pytest.fail("Should not reach here")


def test_run_embedding():
    model = BOOTSTRAPPED_MODEL
    res = model.run_embedding(text=INPUT)
    _assert_is_expected_embedding_result(res)


def test_run_embeddings_str_type():
    """Supposed to be a list, gets fixed automatically."""
    model = BOOTSTRAPPED_MODEL
    res = model.run_embeddings(texts=INPUT)
    assert isinstance(res.results, list)
    assert len(res.results) == 1


def test_run_embeddings():
    model = BOOTSTRAPPED_MODEL
    results = model.run_embeddings(texts=[INPUT])
    assert isinstance(results.results, list)
    _assert_is_expected_embeddings_results(results)


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
def test_run_rerank_query_type_error(query, docs, top_n):
    """test for type checks matching task/run signature"""
    with pytest.raises(TypeError):
        BOOTSTRAPPED_MODEL.run_rerank_query(query=query, documents=docs, top_n=top_n)
        pytest.fail("Should not reach here.")


def test_run_rerank_query_no_type_error():
    """no type error with list of string queries and list of dict documents"""
    BOOTSTRAPPED_MODEL.run_rerank_query(query=QUERY, documents=DOCS, top_n=1)


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
def test_run_rerank_query_top_n(top_n, expected):
    res = BOOTSTRAPPED_MODEL.run_rerank_query(query=QUERY, documents=DOCS, top_n=top_n)
    assert isinstance(res, RerankQueryResult)
    assert len(res.scores) == expected


def test_run_rerank_query_no_query():
    with pytest.raises(TypeError):
        BOOTSTRAPPED_MODEL.run_rerank_query(query=None, documents=DOCS, top_n=99)


def test_run_rerank_query_zero_docs():
    """No empty doc list therefore result is zero result scores"""
    with pytest.raises(ValueError):
        BOOTSTRAPPED_MODEL.run_rerank_query(query=QUERY, documents=[], top_n=99)


def test_run_rerank_query():
    result = BOOTSTRAPPED_MODEL.run_rerank_query(query=QUERY, documents=DOCS)
    assert isinstance(result, RerankQueryResult)

    scores = result.scores
    assert isinstance(scores, list)
    assert len(scores) == len(DOCS)

    types_found = _assert_valid_scores(scores)
    _assert_types_found(types_found)


@pytest.mark.parametrize(
    "queries,docs", [("test string", DOCS), (QUERIES, {"testdict": "not list"})]
)
def test_run_rerank_queries_type_error(queries, docs):
    """type error check ensures params are lists and not just 1 string or just one doc (for example)"""
    with pytest.raises(TypeError):
        BOOTSTRAPPED_MODEL.run_rerank_queries(queries=queries, documents=docs)
        pytest.fail("Should not reach here.")


def test_run_rerank_queries_no_type_error():
    """no type error with list of string queries and list of dict documents"""
    BOOTSTRAPPED_MODEL.run_rerank_queries(queries=QUERIES, documents=DOCS, top_n=99)


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
def test_run_rerank_queries_top_n(top_n, expected):
    """no type error with list of string queries and list of dict documents"""
    res = BOOTSTRAPPED_MODEL.run_rerank_queries(
        queries=QUERIES, documents=DOCS, top_n=top_n
    )
    assert isinstance(res, RerankPredictions)
    assert len(res.results) == len(QUERIES)
    for result in res.results:
        assert len(result.scores) == expected


@pytest.mark.parametrize(
    "queries, docs",
    [
        ([], DOCS),
        (QUERIES, []),
        ([], []),
    ],
    ids=["no queries", "no docs", "no queries and no docs"],
)
def test_run_rerank_queries_no_queries_or_no_docs(queries, docs):
    """No queries and/or no docs therefore result is zero results"""

    with pytest.raises(ValueError):
        BOOTSTRAPPED_MODEL.run_rerank_queries(queries=queries, documents=docs, top_n=9)


def test_run_rerank_queries():
    top_n = 2
    rerank_result = BOOTSTRAPPED_MODEL.run_rerank_queries(
        queries=QUERIES, documents=DOCS, top_n=top_n
    )
    assert isinstance(rerank_result, RerankPredictions)

    results = rerank_result.results
    assert isinstance(results, list)
    assert len(results) == 2 == len(QUERIES)  # 2 queries yields 2 result(s)

    types_found = {}  # Gather the type tests from any of the results

    for result in results:
        assert isinstance(result, RerankQueryResult)
        scores = result.scores
        assert isinstance(scores, list)
        assert len(scores) == top_n
        types_found = _assert_valid_scores(scores, types_found)

    # Make sure our document fields of different types made it in/out ok
    _assert_types_found(types_found)


def test_run_sentence_similarity():
    model = BOOTSTRAPPED_MODEL
    result = model.run_sentence_similarity(source_sentence=QUERY, sentences=SENTENCES)
    scores = result.scores
    assert len(scores) == len(SENTENCES)
    for score in scores:
        assert isinstance(score, float)


def test_run_sentence_similarities():
    model = BOOTSTRAPPED_MODEL
    res = model.run_sentence_similarities(source_sentences=QUERIES, sentences=SENTENCES)
    results = res.results
    assert len(results) == len(QUERIES)
    for result in results:
        scores = result.scores
        assert len(scores) == len(SENTENCES)
        for score in scores:
            assert isinstance(score, float)
