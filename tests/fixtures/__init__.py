"""Helpful fixtures for configuring individual unit tests.
"""
# Standard
from unittest import mock
import os
import random
import tempfile

# Third Party
from datasets import load_dataset
import numpy as np
import pytest
import torch
import transformers

# First Party
from caikit_tgis_backend import TGISBackend
import caikit

# Local
from caikit_nlp.resources.pretrained_model import HFAutoCausalLM, HFAutoSeq2SeqLM
import caikit_nlp

### Constants used in fixtures
FIXTURES_DIR = os.path.join(os.path.dirname(__file__))
TINY_MODELS_DIR = os.path.join(FIXTURES_DIR, "tiny_models")
CAUSAL_LM_MODEL = os.path.join(TINY_MODELS_DIR, "BloomForCausalLM")
SEQ2SEQ_LM_MODEL = os.path.join(TINY_MODELS_DIR, "T5ForConditionalGeneration")


### Stub Modules

# Helper stubs / mocks; we use these to patch caikit so that we don't actually
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


### Fixtures for downloading objects needed to run tests via public internet
@pytest.fixture
def downloaded_dataset(request):
    """Download a dataset via datasets; this should be called via indirect parameterization.

    Example:
    @pytest.mark.parametrize(
        'download_dataset',
        [{"dataset_path": "ought/raft", "dataset_name": "twitter_complaints"}],
        indirect=True
    )
    def test_download_data(downloaded_dataset):
        # downloaded_dataset is a reference to the loaded dataset

    NOTE: Currently this is unused, but we keep it here in case we get some tiny
    models that we want to run quick regression tests on in the future.
    """
    dataset_path = request.param["dataset_path"]
    dataset_name = request.param["dataset_name"]
    return load_dataset(dataset_path, dataset_name)


@pytest.fixture
def temp_cache_dir(request):
    """Use a temporary directory as our transformers / huggingface cache dir. We have permission
    to do things in this directory, but we don't have anything downloaded in it.
    """
    old_cache_path = transformers.utils.hub.TRANSFORMERS_CACHE
    # Create a new tempdir & cache it
    temp_dir = tempfile.TemporaryDirectory()
    transformers.utils.hub.TRANSFORMERS_CACHE = temp_dir.name
    yield
    temp_dir.cleanup()
    if old_cache_path is not None:
        transformers.utils.hub.TRANSFORMERS_CACHE = old_cache_path


@pytest.fixture
def models_cache_dir(request):
    """Use the tiny models directory as the HuggingFace Cache."""
    old_cache_path = transformers.utils.hub.TRANSFORMERS_CACHE
    transformers.utils.hub.TRANSFORMERS_CACHE = TINY_MODELS_DIR
    yield
    if old_cache_path is not None:
        transformers.utils.hub.TRANSFORMERS_CACHE = old_cache_path


### Fixtures for grabbing a randomly initialized model to test interfaces against
## Causal LM
@pytest.fixture
def causal_lm_train_kwargs():
    """Get the kwargs for a valid train call to a Causal LM."""
    model_kwargs = {
        "base_model": HFAutoCausalLM.bootstrap(
            model_name=CAUSAL_LM_MODEL, tokenizer_name=CAUSAL_LM_MODEL
        ),
        "train_stream": caikit.core.data_model.DataStream.from_iterable([]),
        "num_epochs": 0,
        "tuning_config": caikit_nlp.data_model.TuningConfig(
            num_virtual_tokens=8, prompt_tuning_init_text="hello world"
        ),
    }
    return model_kwargs


@pytest.fixture
def causal_lm_dummy_model(causal_lm_train_kwargs):
    """Train a Causal LM dummy model."""
    return caikit_nlp.modules.text_generation.PeftPromptTuning.train(
        **causal_lm_train_kwargs
    )


## Seq2seq
@pytest.fixture
def seq2seq_lm_train_kwargs():
    """Get the kwargs for a valid train call to a Causal LM."""
    model_kwargs = {
        "base_model": HFAutoSeq2SeqLM.bootstrap(
            model_name=SEQ2SEQ_LM_MODEL, tokenizer_name=SEQ2SEQ_LM_MODEL
        ),
        "train_stream": caikit.core.data_model.DataStream.from_iterable([]),
        "num_epochs": 0,
        "tuning_config": caikit_nlp.data_model.TuningConfig(
            num_virtual_tokens=16, prompt_tuning_init_text="hello world"
        ),
    }
    return model_kwargs


@pytest.fixture
def seq2seq_lm_dummy_model(seq2seq_lm_train_kwargs):
    """Train a Seq2Seq LM dummy model."""
    return caikit_nlp.modules.text_generation.PeftPromptTuning.train(
        **seq2seq_lm_train_kwargs
    )


@pytest.fixture()
def requires_determinism(request):
    """Set all of random seeds for tests expected to be deterministic."""
    # Basically, set the random seed of anything our tests might depend on
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


### Args for commonly used datasets that we can use with our fixtures accepting params
TWITTER_DATA_DOWNLOAD_ARGS = [
    {"dataset_path": "ought/raft", "dataset_name": "twitter_complaints"}
]
