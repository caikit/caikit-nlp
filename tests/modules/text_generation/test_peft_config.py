# Standard
from unittest.mock import Mock

# Third Party
import pytest

# Local
from caikit_nlp.data_model import PromptOutputModelType
from caikit_nlp.modules.text_generation.peft_config import TuningType, get_peft_config


@pytest.fixture
def mock_error():
    # Create a mock error object with the expected behavior
    error = Mock()
    error.value_check.side_effect = (
        lambda code, condition, message: None if condition else error(code, message)
    )
    return error


@pytest.fixture
def mock_base_model():
    base_model = Mock()
    base_model.PROMPT_OUTPUT_TYPES = [
        PromptOutputModelType.ENCODER,
        PromptOutputModelType.DECODER,
    ]
    base_model.MAX_NUM_TRANSFORMERS = 2
    return base_model


@pytest.fixture
def mock_cls():
    return Mock()


@pytest.fixture
def mock_torch_dtype():
    return Mock()


@pytest.fixture
def mock_verbalizer():
    return Mock()


@pytest.fixture
def mock_tuning_config():
    # Create a mock tuning_config with a list of output_model_types
    tuning_config = Mock(
        prompt_tuning_init_method="TEXT", prompt_tuning_init_source_model="source_model"
    )
    tuning_config.output_model_types = [
        PromptOutputModelType.ENCODER,
        PromptOutputModelType.DECODER,
    ]

    return tuning_config


def test_get_peft_config(
    mock_error,
    mock_base_model,
    mock_cls,
    mock_torch_dtype,
    mock_verbalizer,
    mock_tuning_config,
):
    # Define some sample values for testing
    tuning_type = TuningType.PROMPT_TUNING

    output_model_types = [PromptOutputModelType.DECODER]

    # Call the function being tested
    task_type, output_model_types, peft_config, tuning_type = get_peft_config(
        tuning_type,
        mock_tuning_config,
        mock_error,
        mock_base_model,
        mock_cls,
        "float32",
        mock_verbalizer,
    )

    # Add assertions to validate the behavior of the function
    assert task_type == mock_base_model.TASK_TYPE
    assert output_model_types == mock_tuning_config.output_model_types
    assert peft_config == mock_cls.create_hf_tuning_config.return_value
    assert tuning_type == TuningType.PROMPT_TUNING

    mock_cls.create_hf_tuning_config.assert_called_once_with(
        base_model=mock_base_model,
        tuning_type=TuningType.PROMPT_TUNING,
        task_type=mock_base_model.TASK_TYPE,
        tokenizer_name_or_path=mock_base_model.model.config._name_or_path,
        tuning_config=mock_tuning_config,
        output_model_types=mock_tuning_config.output_model_types,
    )
