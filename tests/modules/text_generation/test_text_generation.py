"""Tests for text-generation module
"""
# Standard
from unittest import mock
import os
import tempfile

# Third Party
import pytest

# Local
from caikit_nlp.modules.text_generation import TextGeneration
from tests.fixtures import (
    CAUSAL_LM_MODEL,
    SEQ2SEQ_LM_MODEL,
    StubTGISBackend,
    StubTGISClient,
)

SAMPLE_TEXT = "Hello stub"


def test_bootstrap_and_run_causallm():
    """Check if we can bootstrap and run causallm models"""

    model = TextGeneration.bootstrap(CAUSAL_LM_MODEL, load_backend=StubTGISBackend())

    result = model.run(SAMPLE_TEXT, preserve_input_text=True)
    StubTGISClient.validate_unary_generate_response(result)


def test_bootstrap_and_run_seq2seq():
    """Check if we can bootstrap and run seq2seq models"""

    model = TextGeneration.bootstrap(SEQ2SEQ_LM_MODEL, load_backend=StubTGISBackend())

    result = model.run(SAMPLE_TEXT, preserve_input_text=True)
    StubTGISClient.validate_unary_generate_response(result)


def test_run_multi_response_errors():
    """Check if multiple responses errors"""
    with mock.patch.object(StubTGISClient, "Generate") as mock_gen_stream:
        fake_response = mock.Mock()
        fake_response.responses = [mock.Mock(), mock.Mock()]
        mock_gen_stream.return_value = fake_response

        model = TextGeneration.bootstrap(
            SEQ2SEQ_LM_MODEL, load_backend=StubTGISBackend()
        )
        with pytest.raises(ValueError):
            model.run(SAMPLE_TEXT, preserve_input_text=True)


def test_bootstrap_and_save_model():
    """Check if we can bootstrap and save the model successfully"""

    model = TextGeneration.bootstrap(SEQ2SEQ_LM_MODEL)

    with tempfile.TemporaryDirectory() as model_dir:
        model.save(model_dir)
        assert os.path.isfile(os.path.join(model_dir, "config.yml"))


def test_save_model_can_run():
    """Check if the model we bootstrap and save is able to load and run successfully"""
    model = TextGeneration.bootstrap(SEQ2SEQ_LM_MODEL)
    with tempfile.TemporaryDirectory() as model_dir:
        model.save(model_dir)
        del model
        new_model = TextGeneration.load(
            model_dir, load_backend=StubTGISBackend(mock_remote=True)
        )
        result = new_model.run(SAMPLE_TEXT, preserve_input_text=True)
        StubTGISClient.validate_unary_generate_response(result)


def test_remote_tgis_only_model():
    """Make sure that a model can be created and used that will only work with a
    remote TGIS connection (i.e. it has no artifacts)
    """
    model_name = "model-name"
    tgis_backend = StubTGISBackend(mock_remote=True)
    model = TextGeneration(model_name, tgis_backend=tgis_backend)
    with tempfile.TemporaryDirectory() as model_dir:
        model.save(model_dir)
        TextGeneration.load(model_dir, load_backend=tgis_backend)


### Output streaming tests ##############################################################


def test_bootstrap_and_run_stream_out():
    """Check if we can bootstrap and run_stream_out"""
    model = TextGeneration.bootstrap(SEQ2SEQ_LM_MODEL, load_backend=StubTGISBackend())

    stream_result = model.run_stream_out(SAMPLE_TEXT)
    StubTGISClient.validate_stream_generate_response(stream_result)


def test_run_stream_out_with_runtime_error():
    """Check if runtime error from client raises"""

    with mock.patch.object(StubTGISClient, "GenerateStream") as mock_gen_stream:
        mock_gen_stream.side_effect = RuntimeError("An error!")

        model = TextGeneration.bootstrap(
            SEQ2SEQ_LM_MODEL, load_backend=StubTGISBackend()
        )
        with pytest.raises(RuntimeError):
            response = model.run_stream_out(SAMPLE_TEXT, preserve_input_text=True)
            # Need to iterate over stream for error
            next(response)
