"""Tests for CrossEncoderModule"""

# Standard
from typing import List
import os
import tempfile

# Third Party
from pytest import approx
import numpy as np
import pytest

# First Party
from caikit.interfaces.nlp.data_model import (
    RerankResult,
    RerankResults,
    RerankScore,
    RerankScores,
    Token,
    TokenizationResults,
)

# Local
from caikit_nlp.modules.text_embedding import CrossEncoderModule
from tests.fixtures import SEQ_CLASS_MODEL

## Setup ########################################################################

# Bootstrapped sequence classification model for reuse across tests
# .bootstrap is tested separately in the first test
# This model needs a tweak (num_labels = 1) to behave like a cross-encoder.
BOOTSTRAPPED_MODEL = CrossEncoderModule.bootstrap(SEQ_CLASS_MODEL)

# Token counts:
# All expected token counts were calculated with reference to the
# `BertForSequenceClassification` model. Each model's tokenizer behaves differently
# which can lead to the expected token counts being invalid.

INPUT = "The quick brown fox jumps over the lazy dog."
INPUT_TOKEN_COUNT = 36 + 2  # [CLS] Thequickbrownfoxjumpsoverthelazydog. [SEP]

QUERY = "What is foo bar?"
QUERY_TOKEN_COUNT = 13 + 2  # [CLS] Whatisfoobar? [SEP]

QUERIES: List[str] = [
    "Who is foo?",
    "Where is the bar?",
]
QUERIES_TOKEN_COUNT = (9 + 2) + (
    14 + 2
)  # [CLS] Whoisfoo? [SEP], [CLS] Whereisthebar? [SEP]

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

# [CLS] query [SEP] text [SEP] for each text in DOCS.
# Subtract one from QUERY_TOKEN_COUNT to avoid counting
# an extra [SEP].
QUERY_DOCS_TOKENS = (QUERY_TOKEN_COUNT - 1) * len(DOCS) + DOCS_TOKEN_COUNT

# [CLS] query [SEP] text [SEP] for each QUERY for each text in DOCS.
# Subtract len(QUERIES) from QUERY_TOKEN_COUNT to avoid counting
# an extra [SEP].
QUERIES_DOCS_TOKENS = (QUERIES_TOKEN_COUNT - len(QUERIES)) * len(DOCS) + (
    DOCS_TOKEN_COUNT * len(QUERIES)
)


## Tests ########################################################################


@pytest.fixture(scope="module", name="loaded_model")
def fixture_loaded_model(tmp_path_factory):
    models_dir = tmp_path_factory.mktemp("models")
    model_path = str(models_dir / "model_id")
    BOOTSTRAPPED_MODEL.save(model_path)
    model = CrossEncoderModule.load(model_path)
    # Make our tiny test model act more like a cross-encoder model with 1 label
    model.model.config.num_labels = 1
    return model


def _assert_is_expected_scores(rerank_scores):
    # Just testing a few values for readability
    assert isinstance(rerank_scores, RerankScores)
    scores = rerank_scores.scores
    assert approx(scores[0].score) == -0.015608355402946472
    assert approx(scores[1].score) == -0.015612606890499592
    assert approx(scores[2].score) == -0.015648163855075836


def _assert_is_expected_rerank_result(actual):
    assert isinstance(actual, RerankResult)
    scores = actual.result
    _assert_is_expected_scores(scores)


def _assert_is_expected_rerank_results(actual):
    assert isinstance(actual, RerankResults)


def test_bootstrap():
    assert isinstance(
        CrossEncoderModule.bootstrap(SEQ_CLASS_MODEL), CrossEncoderModule
    ), "bootstrap error"


def _assert_valid_scores(scores):
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


def test_bootstrap_model(loaded_model):
    assert isinstance(BOOTSTRAPPED_MODEL, CrossEncoderModule), "bootstrap model type"
    assert (
        BOOTSTRAPPED_MODEL.model.__class__.__name__ == "CrossEncoder"
    ), "bootstrap model class name"
    # worth noting that bootstrap does not wrap, but load does
    assert (
        loaded_model.model.__class__.__name__ == "CrossEncoderWithTruncate"
    ), "loaded model class name"


def test_save_load_and_run():
    """Check if we can load and run a saved model successfully"""
    model_id = "model_id"
    with tempfile.TemporaryDirectory(suffix="-xe-1st") as model_dir:
        model_path = os.path.join(model_dir, model_id)
        BOOTSTRAPPED_MODEL.save(model_path)
        new_model = CrossEncoderModule.load(model_path)

    assert isinstance(new_model, CrossEncoderModule), "save and load error"
    assert new_model != BOOTSTRAPPED_MODEL, "did not load a new model"

    # Make our tiny test model act more like a cross-encoder model
    new_model.model.config.num_labels = 1

    # Use run_rerank_query just to make sure this new model is usable
    top_n = 3
    rerank_result = new_model.run_rerank_query(query=QUERY, documents=DOCS, top_n=top_n)

    assert isinstance(rerank_result, RerankResult)

    result = rerank_result.result
    assert isinstance(result, RerankScores)
    scores = result.scores
    assert isinstance(scores, list)
    assert len(scores) == top_n

    _assert_valid_scores(scores)

    assert rerank_result.input_token_count == QUERY_DOCS_TOKENS
    _assert_is_expected_rerank_result(rerank_result)
    rerank_results = new_model.run_rerank_queries(
        queries=QUERIES, documents=DOCS, top_n=1
    )
    _assert_is_expected_rerank_results(rerank_results)


def test_public_model_info():
    """Check if we can get model info successfully"""
    model_id = "model_id"
    with tempfile.TemporaryDirectory(suffix="-xe-mi") as model_dir:
        model_path = os.path.join(model_dir, model_id)
        BOOTSTRAPPED_MODEL.save(model_path)
        new_model = CrossEncoderModule.load(model_path)

    result = new_model.public_model_info
    assert "max_seq_length" in result
    assert type(result["max_seq_length"]) is int
    assert new_model.model.tokenizer.model_max_length == 512
    assert result["max_seq_length"] == new_model.model.tokenizer.model_max_length

    # We only have the following key(s) in model_info right now for cross-encoders...
    assert list(result.keys()) == ["max_seq_length"]


def test_run_tokenization(loaded_model):
    res = loaded_model.run_tokenizer(text=INPUT)
    assert isinstance(res, TokenizationResults)
    assert isinstance(res.results, list)
    assert isinstance(res.results[0], Token)
    assert res.token_count == INPUT_TOKEN_COUNT


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
    match = r"type check failed"
    with pytest.raises(TypeError, match=match):
        loaded_model.run_rerank_query(query=query, documents=docs, top_n=top_n)
        pytest.fail("Should not reach here.")


@pytest.mark.parametrize("top_n", [1, 99, None])
def test_run_rerank_query_no_type_error(loaded_model, top_n):
    """no type error with list of string queries and list of dict documents"""
    res = loaded_model.run_rerank_query(query=QUERY, documents=DOCS, top_n=top_n)

    # [CLS] query [SEP] text [SEP] for each text in DOCS.
    # Subtract one from QUERY_TOKEN_COUNT to avoid counting
    # an extra [SEP].
    q_tokens = (QUERY_TOKEN_COUNT - 1) * len(DOCS)
    expected = q_tokens + DOCS_TOKEN_COUNT
    assert res.input_token_count == expected


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
    assert res.input_token_count == QUERY_DOCS_TOKENS


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

    _assert_valid_scores(scores)
    assert res.input_token_count == QUERY_DOCS_TOKENS


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

    assert res.input_token_count == QUERIES_DOCS_TOKENS


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
    assert res.input_token_count == QUERIES_DOCS_TOKENS


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

    for result in results:
        assert isinstance(result, RerankScores)
        scores = result.scores
        assert isinstance(scores, list)
        assert len(scores) == top_n
        _assert_valid_scores(scores)

    assert rerank_result.input_token_count == QUERIES_DOCS_TOKENS


@pytest.mark.parametrize("truncate_input_tokens", [-1, 512])
def test_truncate_input_tokens_default(truncate_input_tokens, loaded_model):
    """Test truncation using model max.
    -1 means let the model truncate at its model max
    512 is more explicitly the same thing (this model's max)
    """
    model_max = loaded_model.model.tokenizer.model_max_length

    too_long = "x " * (model_max - 3)  # 3 for tokens (no room for a query token)
    just_barely = "x " * (model_max - 4)  # 3 for tokens plus room for a query token
    queries = ["x"]
    docs = [{"text": t} for t in ["x", too_long, just_barely, too_long, just_barely]]

    # Just testing for no errors raised for now
    _res = loaded_model.run_rerank_queries(
        queries=queries, documents=docs, truncate_input_tokens=truncate_input_tokens
    )


@pytest.mark.parametrize("truncate_input_tokens", [0, 513])
def test_truncate_input_tokens_errors(truncate_input_tokens, loaded_model):
    """Test that we get truncation errors.
    0 (the default) means we return errors when truncation would happen.
    513+ (any number above the max) is treated the same way.
    """
    model_max = loaded_model.model.tokenizer.model_max_length

    too_long = "a " * (model_max - 3)  # 3 for tokens (no room for a query token)
    just_barely = "a " * (model_max - 4)  # 3 for tokens plus room for a query token
    queries = ["q"]

    # Add 50 of these little ones to get past the first batch(es)
    # to verify that this error message index is for the input
    # position and not just an index into some internal batch.
    docs = [{"text": "a"}] * 50
    docs.extend([{"text": t} for t in [too_long, just_barely, too_long, just_barely]])

    match1 = rf"exceeds the maximum sequence length for this model \({model_max}\) for text at indexes: 50, 52."
    with pytest.raises(ValueError, match=match1):
        loaded_model.run_rerank_queries(
            queries=queries, documents=docs, truncate_input_tokens=truncate_input_tokens
        )


@pytest.mark.parametrize("truncate_input_tokens", [-1, 99, 510, 511, 512])
def test_too_many_tokens_with_truncation_working(truncate_input_tokens, loaded_model):
    """truncate_input_tokens prevents these endpoints from raising an error when too many tokens.

    Test with -1 which lets the model do truncation instead of raising an error.
    Test with 99 (< 512 -2) which causes our code to do the truncation instead of raising an error.
    Test with 510 (512 -2) which causes our code to do the truncation instead of raising an error.
    511 and 512 also behave like 510. The value is allowed, but begin/end tokens will take space.
    """

    model_max = loaded_model.model.tokenizer.model_max_length

    ok = "x " * (model_max - 2)  # Subtract 2 for begin/end tokens
    too_long = "x " * (model_max - 1)  # This will go over

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
    "truncate_input_tokens", [1, 2, 3, 4, 5, 6, 99, 100, 101, 510, 511, 512, -1]
)
def test_truncation(truncate_input_tokens, loaded_model):
    """verify that results are as expected with truncation"""

    max_len = loaded_model.model.tokenizer.model_max_length

    if truncate_input_tokens is None or truncate_input_tokens < 0:
        # For -1 we don't truncate, but model will
        repeat = max_len
    else:
        repeat = min(
            truncate_input_tokens, max_len
        )  # max_len is used when we need -4 for begin/"q"/sep/end

    # Build a text like "x x x.. x " with room for one more token
    repeat = repeat - 4  # room for separators and a single-token query
    repeat = repeat - 1  # space for the final x or y token to show difference

    base = ""
    if repeat > 0:
        base = "x " * repeat  # A bunch of "x" tokens
    x = base + "x"  # One last "x" that will not get truncated
    y = base + "y"  # A different last character "y" not truncated
    z = y + " z"  # Add token "z" after "y". This should get truncated.

    # Multiple queries to test query-loop vs queries
    # Query for the significant added chars to affect score.
    queries = ["y", "z"]
    docs = [{"text": t} for t in [x, y, z]]
    res = loaded_model.run_rerank_queries(
        queries=queries,
        documents=docs,
        truncate_input_tokens=truncate_input_tokens,
    )
    queries_results = res.results

    # Compare with results from individual embedding calls in a loop
    query_results = []
    for query in queries:
        r = loaded_model.run_rerank_query(
            query=query,
            documents=docs,
            truncate_input_tokens=truncate_input_tokens,
        )
        query_results.append(r.result)

    assert len(queries_results) == len(
        query_results
    ), "expected the same length results"

    # compare the scores (queries call vs query call in a loop)
    # order is the same
    for i, r in enumerate(queries_results):
        queries_scores = [x.score for x in r.scores]
        query_scores = [x.score for x in query_results[i].scores]
        assert np.array_equal(queries_scores, query_scores)
        # To compare scores based on the inputs, we need to use the index too
        indexed_query_scores = {s.index: s.score for s in query_results[i].scores}

        # Make sure the x...xx, x...xy are not a match (we kept the significant last token)
        assert indexed_query_scores[0] != indexed_query_scores[1]

        # x...xy is the same as x...xyz because we truncated the z token -- it worked!
        assert indexed_query_scores[1] == indexed_query_scores[2]
