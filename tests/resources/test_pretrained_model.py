"""Tests for all pretrained model resource types; since most of the functionality for
these classes live through the base class, we test them all in this file.

Importantly, these tests do NOT depend on access to external internet to run correctly.

NOTE: If the subclasses become sufficiently complex, this test file should be split up.
"""
# Standard
from unittest.mock import patch

# Third Party
from transformers import AutoModelForCausalLM
import pytest
import torch

# First Party
import aconfig

# Local
from caikit_nlp.resources.pretrained_model import HFAutoCausalLM
from tests.fixtures import CAUSAL_LM_MODEL, models_cache_dir, temp_cache_dir


def test_boostrap_causal_lm(models_cache_dir):
    """Ensure that we can bootstrap a causal LM if we have download access."""
    # If we have an empty cachedir & do allow downloads, we should be able to init happily
    base_model = HFAutoCausalLM.bootstrap(
        model_name=CAUSAL_LM_MODEL, tokenizer_name=CAUSAL_LM_MODEL
    )
    assert isinstance(base_model, HFAutoCausalLM)
    assert base_model.MODEL_TYPE is AutoModelForCausalLM
    assert base_model.TASK_TYPE == "CAUSAL_LM"


def test_boostrap_causal_lm_override_dtype(models_cache_dir):
    """Ensure that we can override the data type to init the model with at bootstrap time."""
    # If we have an empty cachedir & do allow downloads, we should be able to init happily,
    # and if we provide a data type override, that override should be applied at bootstrap time.
    base_model = HFAutoCausalLM.bootstrap(
        model_name=CAUSAL_LM_MODEL,
        tokenizer_name=CAUSAL_LM_MODEL,
        torch_dtype=torch.float64,
    )
    assert isinstance(base_model, HFAutoCausalLM)
    assert base_model.MODEL_TYPE is AutoModelForCausalLM
    assert base_model.TASK_TYPE == "CAUSAL_LM"
    assert base_model.model.dtype is torch.float64


@patch("transformers.models.auto.tokenization_auto.AutoTokenizer.from_pretrained")
def test_boostrap_causal_lm_download_disabled(mock_tok_from_pretrained, temp_cache_dir):
    """Ensure that we can't try to download if downloads are disabled"""
    # NOTE: allow_downloads is false by default
    mock_tok_from_pretrained.side_effect = RuntimeError("It's a mock")
    with pytest.raises(RuntimeError):
        HFAutoCausalLM.bootstrap(model_name="foo/bar/baz", tokenizer_name="foo/bar/baz")
    kwargs = mock_tok_from_pretrained.call_args[1]
    assert kwargs["local_files_only"] == True


@patch("transformers.models.auto.tokenization_auto.AutoTokenizer.from_pretrained")
def test_boostrap_causal_lm_download_enabled(mock_tok_from_pretrained, temp_cache_dir):
    """Ensure that we can try to download if downloads are enabled."""
    with patch(
        "caikit_nlp.resources.pretrained_model.base.get_config",
        return_value=aconfig.Config({"allow_downloads": True}),
    ):
        mock_tok_from_pretrained.side_effect = RuntimeError("It's a mock")
        with pytest.raises(RuntimeError):
            HFAutoCausalLM.bootstrap(
                model_name="foo/bar/baz", tokenizer_name="foo/bar/baz"
            )
        kwargs = mock_tok_from_pretrained.call_args[1]
        assert kwargs["local_files_only"] == False
