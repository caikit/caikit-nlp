"""Tests for prompt tuning based inference via TGIS backend; note that these tests mock the
TGIS client and do NOT actually start/hit a TGIS server instance.
"""
# Standard
from unittest import mock
from unittest.mock import patch
import os
import tempfile

# Third Party
import pytest

# Local
from caikit_pt.blocks.text_generation import PeftPromptTuningTGIS
from tests.fixtures import causal_lm_dummy_model, causal_lm_train_kwargs


### Helper stubs / mocks; we use these to patch caikit so that we don't actually
# test the TGIS backend directly, and instead stub the client and inspect the
# args that we pass to it.
class StubClient:
    def __init__(self, base_model_name):
        pass

    # Generation calls on this class are a mock that explodes when invoked
    Generate = mock.Mock(side_effect=RuntimeError("TGIS client is a mock!"))


class StubBackend:
    def get_client(self, base_model_name):
        return StubClient(base_model_name)


@patch("caikit.core.module_backend_config.get_backend")
def test_load_and_run(mock_get_backend, causal_lm_dummy_model):
    """Ensure we can export an in memory model, load it, and (mock) run it with the right text & prefix ID."""
    # Patch our stub backend into caikit so that we don't actually try to start TGIS
    mock_get_backend.return_value = StubBackend()
    causal_lm_dummy_model.verbalizer = "hello distributed {{input}}"

    # Save the local model & reload it a TGIS backend distributed module
    # Also, save the name of the dir + prompt ID, which is the path TGIS expects for the prefix ID
    with tempfile.TemporaryDirectory() as model_dir:
        causal_lm_dummy_model.save(model_dir)
        mock_tgis_model = PeftPromptTuningTGIS.load(model_dir)
        model_prompt_dir = os.path.split(model_dir)[-1]

    # Run an inference request, which is wrapped around our mocked Generate call
    sample_text = "Hello stub"
    with pytest.raises(RuntimeError):
        mock_tgis_model.run(sample_text, preserve_input_text=True)
    assert len(StubClient.Generate.call_args_list) == 1
    stub_generation_request = StubClient.Generate.call_args_list[0].args[0]

    # Validate that our verbalizer carried over correctly & was applied at inference time
    assert mock_tgis_model.verbalizer == causal_lm_dummy_model.verbalizer
    assert stub_generation_request.requests[0].text == "hello distributed {}".format(
        sample_text
    )

    # Ensure that our prefix ID matches the expected path based on our tmpdir and config
    assert model_prompt_dir == stub_generation_request.prefix_id
