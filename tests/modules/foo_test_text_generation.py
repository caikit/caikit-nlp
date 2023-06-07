"""Tests for text-generation module
"""
# Standard
from unittest import mock
from unittest.mock import patch
import os
import tempfile

# Third Party
import pytest

# Local
from caikit_nlp.blocks.text_generation import TextGeneration
from caikit_nlp.data_model.generation import GeneratedResult

from tests.fixtures import CAUSAL_LM_MODEL, SEQ2SEQ_LM_MODEL

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
def test_bootstrap_and_run_causallm(mock_get_backend, causal_lm_dummy_model):
    """Check if we can bootstrap and run causallm models"""
    # Patch our stub backend into caikit so that we don't actually try to start TGIS
    mock_get_backend.return_value = StubBackend()

    model = TextGeneration.bootstrap(CAUSAL_LM_MODEL)

    sample_text = "Hello stub"
    response = model.run(sample_text, preserve_input_text=True)

    assert isinstance(response, GeneratedResult)



