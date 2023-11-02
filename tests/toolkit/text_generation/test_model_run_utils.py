# Standard
from transformers import T5ForConditionalGeneration, AutoTokenizer

# First Party
from caikit.interfaces.nlp.data_model import (
    GeneratedTextResult
)

# Local
from tests.fixtures import (
    SEQ2SEQ_LM_MODEL
)
from caikit_nlp.toolkit.text_generation.model_run_utils import generate_text_func

model = T5ForConditionalGeneration.from_pretrained(SEQ2SEQ_LM_MODEL)
tokenizer = AutoTokenizer.from_pretrained(SEQ2SEQ_LM_MODEL, model_max_length=512)

def test_generate_text():
    train_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "producer_id": 0,
        "eos_token": "<\n>",
        "text": "I work at IBM"
    }

    pred = generate_text_func(**train_kwargs)
    assert isinstance(pred, GeneratedTextResult)