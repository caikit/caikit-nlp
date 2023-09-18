# Standard
from unittest.mock import Mock

# Third Party
from peft import PromptTuningConfig
import pytest

# Local
from caikit_nlp.data_model import PromptOutputModelType, TuningConfig
from caikit_nlp.modules.text_generation.peft_config import TuningType, get_peft_config
from tests.fixtures import (
    seq2seq_lm_dummy_model,
    seq2seq_lm_train_kwargs,
)


def test_get_peft_config(
    seq2seq_lm_train_kwargs,
    seq2seq_lm_dummy_model,
):
    # Define some sample values for testing
    tuning_type = TuningType.PROMPT_TUNING
    tuning_config = TuningConfig(
        num_virtual_tokens=8,
        prompt_tuning_init_method="TEXT",
        prompt_tuning_init_text="Hello world"
    )
    output_model_types = [PromptOutputModelType.DECODER]
    stub_base_model = seq2seq_lm_train_kwargs["base_model"]
    # Call the function being tested
    task_type, output_model_types, peft_config, tuning_type = get_peft_config(
        tuning_type,
        tuning_config,
        stub_base_model,
        seq2seq_lm_dummy_model,
        "float32",
        "{{input}}",
    )

    # Add assertions to validate the behavior of the function
    assert task_type == stub_base_model.TASK_TYPE
    assert output_model_types == stub_base_model.PROMPT_OUTPUT_TYPES
    assert tuning_type == TuningType.PROMPT_TUNING

    # Validation for type & important fields in the peft config
    assert isinstance(peft_config, PromptTuningConfig)
    assert peft_config.num_virtual_tokens == tuning_config.num_virtual_tokens
    assert peft_config.task_type == stub_base_model.TASK_TYPE
    assert peft_config.prompt_tuning_init == tuning_config.prompt_tuning_init_method
    assert peft_config.prompt_tuning_init_text == tuning_config.prompt_tuning_init_text
