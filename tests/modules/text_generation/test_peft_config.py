# Standard
from unittest.mock import Mock

# Third Party
from peft import PromptTuningConfig
import pytest

# Local
from caikit_nlp.data_model import TuningConfig
from caikit_nlp.modules.text_generation import TextGeneration
from caikit_nlp.modules.text_generation.peft_config import (
    TuningType,
    get_peft_config,
    resolve_base_model,
)
from caikit_nlp.resources.pretrained_model import HFAutoSeq2SeqLM
from tests.fixtures import (
    SEQ2SEQ_LM_MODEL,
    TINY_MODELS_DIR,
    causal_lm_dummy_model,
    causal_lm_train_kwargs,
    seq2seq_lm_dummy_model,
    seq2seq_lm_train_kwargs,
    temp_config,
)


@pytest.mark.parametrize(
    "train_kwargs,dummy_model",
    [
        (
            "seq2seq_lm_train_kwargs",
            "seq2seq_lm_dummy_model",
        ),
        ("causal_lm_train_kwargs", "causal_lm_dummy_model"),
    ],
)
def test_get_peft_config(train_kwargs, dummy_model, request):
    # Fixtures can't be called directly or passed to mark parametrize;
    # Currently, passing the fixture by name and retrieving it through
    # the request is the 'right' way to do this.
    train_kwargs = request.getfixturevalue(train_kwargs)
    dummy_model = request.getfixturevalue(dummy_model)

    # Define some sample values for testing
    tuning_type = TuningType.PROMPT_TUNING
    tuning_config = TuningConfig(
        num_virtual_tokens=8,
        prompt_tuning_init_method="TEXT",
        prompt_tuning_init_text="Hello world",
    )
    dummy_resource = train_kwargs["base_model"]

    # Call the function being tested
    task_type, output_model_types, peft_config, tuning_type = get_peft_config(
        tuning_type,
        tuning_config,
        dummy_resource,
        dummy_model,
        "float32",
        "{{input}}",
    )

    # Add assertions to validate the behavior of the function
    assert task_type == dummy_resource.TASK_TYPE
    assert output_model_types == dummy_resource.PROMPT_OUTPUT_TYPES
    assert tuning_type == TuningType.PROMPT_TUNING

    # Validation for type & important fields in the peft config
    assert isinstance(peft_config, PromptTuningConfig)
    assert peft_config.num_virtual_tokens == tuning_config.num_virtual_tokens
    assert peft_config.task_type == dummy_resource.TASK_TYPE
    assert peft_config.prompt_tuning_init == tuning_config.prompt_tuning_init_method
    assert peft_config.prompt_tuning_init_text == tuning_config.prompt_tuning_init_text


def test_resolve_model_with_invalid_path_raises():
    """Test passing invalid path to resolve_model function raises"""

    invalid_base_model = "path/../../important"
    with pytest.raises(ValueError):
        resolve_base_model(invalid_base_model, None, "foo")


def test_resolve_model_with_valid_folder_path():
    """Test passing valid folder path to resolve_model function works"""

    model = resolve_base_model(SEQ2SEQ_LM_MODEL, TextGeneration, "float32")

    assert isinstance(model, HFAutoSeq2SeqLM)


def test_resolve_model_works_preloaded_model():

    base_model = HFAutoSeq2SeqLM.bootstrap(SEQ2SEQ_LM_MODEL)
    resolved_model = resolve_base_model(base_model, TextGeneration, "float32")
    assert isinstance(resolved_model, HFAutoSeq2SeqLM)


def test_resolve_model_with_different_base_path_works():

    base_model_name = "T5ForConditionalGeneration"
    with temp_config(base_models_dir=TINY_MODELS_DIR):
        resolved_model = resolve_base_model(base_model_name, TextGeneration, "float32")
        assert isinstance(resolved_model, HFAutoSeq2SeqLM)
