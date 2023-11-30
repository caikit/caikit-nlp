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


@pytest.mark.parametrize("causal_model_fixture", ["causal_lm_dummy_model"])
def test_generate_text_func_preserve_input_causal_lm(request, causal_model_fixture):
    """For Causal LM task types, setting preserve_inout_text to True
    will result in input text in model prediction. Setting to False will
    strip the input text from model prediction.
    """
    input_text = "What is the boiling point of liquid Nitrogen?"
    causal_model = request.getfixturevalue(causal_model_fixture)
    # assert type(causal_model.model) == False
    generated_text = generate_text_func(
        model=causal_model.model,
        tokenizer=causal_model.tokenizer,
        producer_id=ProducerId("TextGeneration", "0.1.0"),
        eos_token="<\n>",
        text=input_text,
        preserve_input_text=True,
        task_type="CAUSAL_LM",
    )
    assert input_text in generated_text.generated_text
    generated_text = generate_text_func(
        model=causal_model.model,
        tokenizer=causal_model.tokenizer,
        producer_id=ProducerId("TextGeneration", "0.1.0"),
        eos_token="<\n>",
        text=input_text,
        preserve_input_text=False,
        task_type="CAUSAL_LM",
    )
    assert input_text not in generated_text.generated_text


@pytest.mark.parametrize("seq_model_fixture", ["seq2seq_lm_dummy_model"])
def test_generate_text_func_preserve_input(request, seq_model_fixture):
    """For Seq2Seq LM task types, setting preserve_inout_text to True
    or False should not change predictions.
    """
    input_text = "What is the boiling point of liquid Nitrogen?"
    seq_model = request.getfixturevalue(seq_model_fixture)
    # assert type(causal_model.model) == False
    generated_text = generate_text_func(
        model=seq_model.model,
        tokenizer=seq_model.tokenizer,
        producer_id=ProducerId("TextGeneration", "0.1.0"),
        eos_token="<\n>",
        text=input_text,
        preserve_input_text=True,
        task_type="SEQ_2_SEQ_LM",
    )
    before_pred = generated_text.generated_text
    generated_text = generate_text_func(
        model=seq_model.model,
        tokenizer=seq_model.tokenizer,
        producer_id=ProducerId("TextGeneration", "0.1.0"),
        eos_token="<\n>",
        text=input_text,
        preserve_input_text=False,
        task_type="SEQ_2_SEQ_LM",
    )
    after_pred = generated_text.generated_text
    assert before_pred == after_pred
