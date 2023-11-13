# Third Party
import pytest

# First Party
from caikit.core.data_model.producer import ProducerId
from caikit.interfaces.nlp.data_model import GeneratedTextResult

# Local
from caikit_nlp.toolkit.text_generation.model_run_utils import generate_text_func
from tests.fixtures import (
    causal_lm_dummy_model,
    causal_lm_train_kwargs,
    seq2seq_lm_dummy_model,
    seq2seq_lm_train_kwargs,
)


@pytest.mark.parametrize(
    "model_fixture", ["seq2seq_lm_dummy_model", "causal_lm_dummy_model"]
)
@pytest.mark.parametrize(
    "serialization_method,expected_type",
    [
        ("to_dict", dict),
        ("to_json", str),
        ("to_proto", GeneratedTextResult._proto_class),
    ],
)
def test_generate_text_func_serialization_json(
    request,
    model_fixture,
    serialization_method,
    expected_type,
):
    model = request.getfixturevalue(model_fixture)
    generated_text = generate_text_func(
        model=model.model,
        tokenizer=model.tokenizer,
        producer_id=ProducerId("TextGeneration", "0.1.0"),
        eos_token="<\n>",
        text="What is the boiling point of liquid Nitrogen?",
    )

    serialized = getattr(generated_text, serialization_method)()
    assert isinstance(serialized, expected_type)
