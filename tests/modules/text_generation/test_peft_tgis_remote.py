"""Tests for prompt tuning based inference via TGIS backend; note that these tests mock the
TGIS client and do NOT actually start/hit a TGIS server instance.
"""
# Standard
from typing import Iterable
from unittest import mock
import os
import tempfile

# Third Party
import pytest

# First Party
from caikit.interfaces.nlp.data_model import GeneratedTextResult
from caikit_tgis_backend import TGISBackend

# Local
from caikit_nlp.modules.text_generation import PeftPromptTuningTGIS
from tests.fixtures import causal_lm_dummy_model, causal_lm_train_kwargs

### Stub Modules

SAMPLE_TEXT = "Hello stub"

# Helper stubs / mocks; we use these to patch caikit so that we don't actually
# test the TGIS backend directly, and instead stub the client and inspect the
# args that we pass to it.
class StubClient:
    def __init__(self, base_model_name):
        pass

    def Generate(self, request):
        return StubClient.unary_generate(request)

    def GenerateStream(self, request):
        return StubClient.stream_generate(request)

    @staticmethod
    def unary_generate(request):
        fake_response = mock.Mock()
        fake_result = mock.Mock()
        fake_result.stop_reason = 5
        fake_result.generated_token_count = 1
        fake_result.text = "moose"
        fake_response.responses = [fake_result]
        return fake_response

    @staticmethod
    def stream_generate(request):
        fake_stream = mock.Mock()
        fake_stream.stop_reason = 5
        fake_stream.generated_token_count = 1
        fake_stream.seed = 10
        fake_stream.text = "moose"
        for _ in range(3):
            yield fake_stream


class StubBackend(TGISBackend):
    def get_client(self, base_model_name):
        return StubClient(base_model_name)


def test_load_and_run(causal_lm_dummy_model):
    """Ensure we can export an in memory model, load it, and (mock) run it with the right text & prefix ID."""
    # Patch our stub backend into caikit so that we don't actually try to start TGIS
    causal_lm_dummy_model.verbalizer = "hello distributed {{input}}"

    with mock.patch.object(StubClient, "Generate") as mock_gen:
        mock_gen.side_effect = StubClient.unary_generate

        # Save the local model & reload it a TGIS backend distributed module
        # Also, save the name of the dir + prompt ID, which is the path TGIS expects for the prefix ID
        with tempfile.TemporaryDirectory() as model_dir:
            causal_lm_dummy_model.save(model_dir)
            mock_tgis_model = PeftPromptTuningTGIS.load(model_dir, StubBackend())
            model_prompt_dir = os.path.split(model_dir)[-1]

        # Run an inference request, which is wrapped around our mocked Generate call
        result = mock_tgis_model.run(SAMPLE_TEXT, preserve_input_text=True)
        assert isinstance(result, GeneratedTextResult)
        assert result.generated_text == "moose"
        assert result.generated_tokens == 1
        assert result.finish_reason == 5

        stub_generation_request = mock_gen.call_args_list[0].args[0]

        # Validate that our verbalizer carried over correctly & was applied at inference time
        assert mock_tgis_model.verbalizer == causal_lm_dummy_model.verbalizer
        assert stub_generation_request.requests[
            0
        ].text == "hello distributed {}".format(SAMPLE_TEXT)

        # Ensure that our prefix ID matches the expected path based on our tmpdir and config
        assert model_prompt_dir == stub_generation_request.prefix_id


def test_load_and_run_stream_out(causal_lm_dummy_model):
    """Ensure we can export an in memory model, load it, and (mock) run output streaming
    with the right text & prefix ID."""
    # Patch our stub backend into caikit so that we don't actually try to start TGIS
    causal_lm_dummy_model.verbalizer = "hello distributed {{input}}"

    with mock.patch.object(StubClient, "GenerateStream") as mock_gen:
        mock_gen.side_effect = StubClient.stream_generate

        # Save the local model & reload it a TGIS backend distributed module
        # Also, save the name of the dir + prompt ID, which is the path TGIS expects for the prefix ID
        with tempfile.TemporaryDirectory() as model_dir:
            causal_lm_dummy_model.save(model_dir)
            mock_tgis_model = PeftPromptTuningTGIS.load(model_dir, StubBackend())
            model_prompt_dir = os.path.split(model_dir)[-1]

        # Run an inference request, which is wrapped around our mocked GenerateStream call
        stream_result = mock_tgis_model.run_stream_out(
            SAMPLE_TEXT, preserve_input_text=True
        )
        assert isinstance(stream_result, Iterable)
        # Convert to list to more easily check outputs
        result_list = list(stream_result)
        assert len(result_list) == 3
        first_result = result_list[0]
        assert first_result.generated_text == "moose"
        assert first_result.details.finish_reason == 5
        assert first_result.details.generated_tokens == 1
        assert first_result.details.seed == 10

        stub_generation_request = mock_gen.call_args_list[0].args[0]

        # Validate that our verbalizer carried over correctly & was applied at inference time
        assert mock_tgis_model.verbalizer == causal_lm_dummy_model.verbalizer
        assert stub_generation_request.request.text == "hello distributed {}".format(
            SAMPLE_TEXT
        )

        # Ensure that our prefix ID matches the expected path based on our tmpdir and config
        assert model_prompt_dir == stub_generation_request.prefix_id
