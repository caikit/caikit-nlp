"""Tests for text-generation module
"""
# Standard
from unittest import mock
from unittest.mock import patch
import os
import tempfile

# Third Party
import pytest

# First Party
from caikit_tgis_backend import TGISBackend

# Local
from caikit_nlp.modules.text_generation import TextGeneration
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


class StubBackend(TGISBackend):
    def get_client(self, base_model_name):
        return StubClient(base_model_name)


def test_bootstrap_and_run_causallm():
    """Check if we can bootstrap and run causallm models"""

    model = TextGeneration.bootstrap(CAUSAL_LM_MODEL, load_backend=StubBackend())

    sample_text = "Hello stub"
    with pytest.raises(RuntimeError):
        model.run(sample_text, preserve_input_text=True)


def test_bootstrap_and_run_seq2seq():
    """Check if we can bootstrap and run seq2seq models"""

    model = TextGeneration.bootstrap(SEQ2SEQ_LM_MODEL, load_backend=StubBackend())

    sample_text = "Hello stub"
    with pytest.raises(RuntimeError):
        model.run(sample_text, preserve_input_text=True)

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
        new_model = TextGeneration.load(model_dir, load_backend=StubBackend())
        sample_text = "Hello stub"
        with pytest.raises(RuntimeError):
            new_model.run(sample_text, preserve_input_text=True)
