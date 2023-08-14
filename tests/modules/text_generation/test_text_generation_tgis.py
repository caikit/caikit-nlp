"""Tests for text-generation module
"""
# Standard
from unittest import mock
import os
import platform
import tempfile

# Third Party
import pytest
import torch

# First Party
from caikit.interfaces.nlp.data_model import GeneratedTextResult
import caikit

# Local
from caikit_nlp.data_model.generation import GenerationTrainRecord
from caikit_nlp.modules.text_generation import TextGeneration, TextGenerationTGIS
from caikit_nlp.resources.pretrained_model.hf_auto_seq2seq_lm import HFAutoSeq2SeqLM
from tests.fixtures import (
    CAUSAL_LM_MODEL,
    SEQ2SEQ_LM_MODEL,
    StubTGISBackend,
    StubTGISClient,
)

SAMPLE_TEXT = "Hello stub"


def test_bootstrap_and_run_causallm():
    """Check if we can bootstrap and run causallm models"""

    model = TextGenerationTGIS.bootstrap(
        CAUSAL_LM_MODEL, load_backend=StubTGISBackend()
    )

    result = model.run(SAMPLE_TEXT, preserve_input_text=True)
    StubTGISClient.validate_unary_generate_response(result)


def test_bootstrap_and_run_seq2seq():
    """Check if we can bootstrap and run seq2seq models"""

    model = TextGenerationTGIS.bootstrap(
        SEQ2SEQ_LM_MODEL, load_backend=StubTGISBackend()
    )

    result = model.run(SAMPLE_TEXT, preserve_input_text=True)
    StubTGISClient.validate_unary_generate_response(result)


def test_run_multi_response_errors():
    """Check if multiple responses errors"""
    with mock.patch.object(StubTGISClient, "Generate") as mock_gen_stream:
        fake_response = mock.Mock()
        fake_response.responses = [mock.Mock(), mock.Mock()]
        mock_gen_stream.return_value = fake_response

        model = TextGenerationTGIS.bootstrap(
            SEQ2SEQ_LM_MODEL, load_backend=StubTGISBackend()
        )
        with pytest.raises(ValueError):
            model.run(SAMPLE_TEXT, preserve_input_text=True)


def test_bootstrap_and_save_model():
    """Check if we can bootstrap and save the model successfully"""

    model = TextGenerationTGIS.bootstrap(
        SEQ2SEQ_LM_MODEL, load_backend=StubTGISBackend()
    )

    with tempfile.TemporaryDirectory() as model_dir:
        model.save(model_dir)
        assert os.path.isfile(os.path.join(model_dir, "config.yml"))


def test_save_model_can_run():
    """Check if the model we bootstrap and save is able to load and run successfully"""
    model = TextGenerationTGIS.bootstrap(SEQ2SEQ_LM_MODEL)
    with tempfile.TemporaryDirectory() as model_dir:
        model.save(model_dir)
        del model
        new_model = TextGenerationTGIS.load(
            model_dir, load_backend=StubTGISBackend(mock_remote=True)
        )
        result = new_model.run(SAMPLE_TEXT, preserve_input_text=True)
        StubTGISClient.validate_unary_generate_response(result)


@pytest.mark.skipif(platform.processor() == "arm", reason="ARM training not supported")
def test_local_train_load_tgis():
    """Check if the model trained in local module is able to
    be loaded in TGIS module / backend
    """
    train_kwargs = {
        "base_model": HFAutoSeq2SeqLM.bootstrap(
            model_name=SEQ2SEQ_LM_MODEL, tokenizer_name=SEQ2SEQ_LM_MODEL
        ),
        "num_epochs": 1,
        "train_stream": caikit.core.data_model.DataStream.from_iterable(
            [
                GenerationTrainRecord(
                    input="@foo what a cute dog!", output="no complaint"
                )
            ]
        ),
        "torch_dtype": torch.float32,
    }
    model = TextGeneration.train(**train_kwargs)
    with tempfile.TemporaryDirectory() as model_dir:
        model.save(model_dir)
        new_model = TextGenerationTGIS.load(
            model_dir, load_backend=StubTGISBackend(mock_remote=True)
        )
        sample_text = "Hello stub"
        generated_text = new_model.run(sample_text)
        assert isinstance(generated_text, GeneratedTextResult)


def test_remote_tgis_only_model():
    """Make sure that a model can be created and used that will only work with a
    remote TGIS connection (i.e. it has no artifacts)
    """
    model_name = "model-name"
    tgis_backend = StubTGISBackend(mock_remote=True)
    model = TextGenerationTGIS(model_name, tgis_backend=tgis_backend)
    with tempfile.TemporaryDirectory() as model_dir:
        model.save(model_dir)
        TextGenerationTGIS.load(model_dir, load_backend=tgis_backend)


### Output streaming tests ##############################################################


def test_bootstrap_and_run_stream_out():
    """Check if we can bootstrap and run_stream_out"""
    model = TextGenerationTGIS.bootstrap(
        SEQ2SEQ_LM_MODEL, load_backend=StubTGISBackend()
    )

    stream_result = model.run_stream_out(SAMPLE_TEXT)
    StubTGISClient.validate_stream_generate_response(stream_result)


def test_run_stream_out_with_runtime_error():
    """Check if runtime error from client raises"""

    with mock.patch.object(StubTGISClient, "GenerateStream") as mock_gen_stream:
        mock_gen_stream.side_effect = RuntimeError("An error!")

        model = TextGenerationTGIS.bootstrap(
            SEQ2SEQ_LM_MODEL, load_backend=StubTGISBackend()
        )
        with pytest.raises(RuntimeError):
            response = model.run_stream_out(SAMPLE_TEXT, preserve_input_text=True)
            # Need to iterate over stream for error
            next(response)
