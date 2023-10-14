"""Tests for sequence classification module
"""
# Standard
from typing import List
import os
import tempfile

# Third Party
import pytest

# Local
from caikit_nlp import RerankQueryResult, RerankScore
from caikit_nlp.data_model import RerankPrediction
from caikit_nlp.modules.reranker import Rerank
from tests.fixtures import SEQ_CLASS_MODEL

## Setup ########################################################################

# Bootstrapped sequence classification model for reusability across tests
# .bootstrap is tested separately in the first test
BOOTSTRAPPED_MODEL = Rerank.bootstrap(SEQ_CLASS_MODEL)

QUERIES: List[str] = [
    "Who is foo?",
    "Where is the bar?",
]

DOCS = [
    {
        "text": "foo",
        "title": "title or whatever",
        "str_test": "test string",
        "int_test": 1,
        "float_test": 1.11,
        "score": 99999,
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

## Tests ########################################################################


def test_bootstrap():
    assert isinstance(BOOTSTRAPPED_MODEL, Rerank), "bootstrap error"


@pytest.mark.parametrize(
    "queries,docs", [("test string", DOCS), (QUERIES, {"testdict": "not list"})]
)
def test_run_type_error(queries, docs):
    """type error check ensures params are lists and not just 1 string or just one doc (for example)"""
    with pytest.raises(TypeError):
        BOOTSTRAPPED_MODEL.run(queries=queries, documents=docs)
        pytest.fail("Should not reach here.")


def test_run_no_type_error():
    """no type error with list of string queries and list of dict documents"""
    BOOTSTRAPPED_MODEL.run(queries=QUERIES, documents=DOCS)


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
def test_run_top_n(top_n, expected):
    """no type error with list of string queries and list of dict documents"""
    res = BOOTSTRAPPED_MODEL.run(queries=QUERIES, documents=DOCS, top_n=top_n)
    assert isinstance(res, RerankPrediction)
    assert len(res.results) == len(QUERIES)
    for result in res.results:
        assert len(result.scores) == expected


def test_save_and_load_and_run_model():
    """Save and load and run a model"""

    model_id = "model_id"
    with tempfile.TemporaryDirectory(suffix="-1st") as model_dir:
        model_path = os.path.join(model_dir, model_id)
        BOOTSTRAPPED_MODEL.save(model_path)
        new_model = Rerank.load(model_path)

    assert isinstance(new_model, Rerank), "save and load error"
    assert new_model != BOOTSTRAPPED_MODEL, "did not load a new model"

    top_n = 2
    rerank_result = new_model.run(queries=QUERIES, documents=DOCS, top_n=top_n)
    assert isinstance(rerank_result, RerankPrediction)

    results = rerank_result.results
    assert isinstance(results, list)
    assert len(results) == 2 == len(QUERIES)  # 2 queries yields 2 result(s)

    # Collect some of the pass-through extras to verify we can do some types
    str_test = None
    int_test = None
    float_test = None

    for result in results:
        assert isinstance(result, RerankQueryResult)
        scores = result.scores
        assert isinstance(scores, list)
        assert len(scores) == top_n
        for score in scores:
            assert isinstance(score, RerankScore)
            assert isinstance(score.score, float)
            assert isinstance(score.corpus_id, int)
            assert score.document == DOCS[score.corpus_id]

            # Test pass-through score (None or 9999) is independent of the result score
            assert score.score != score.document.get(
                "score"
            ), "unexpected passthru score same as result score"

            # Gather various type test values
            str_test = score.document.get("str_test", str_test)
            int_test = score.document.get("int_test", int_test)
            float_test = score.document.get("float_test", float_test)

    assert type(str_test) == str, "passthru str value type check"
    assert type(int_test) == int, "passthru int value type check"
    assert type(float_test) == float, "passthru float value type check"
