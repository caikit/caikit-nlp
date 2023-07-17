"""Tests for text-generation module
"""
# Standard
import os
import tempfile

# Local
from caikit_nlp.data_model import GeneratedResult
from caikit_nlp.modules.text_generation import TextGeneration
from tests.fixtures import CAUSAL_LM_MODEL, SEQ2SEQ_LM_MODEL

### Stub Modules


def test_bootstrap_and_run_causallm():
    """Check if we can bootstrap and run causallm models"""

    model = TextGeneration.bootstrap(CAUSAL_LM_MODEL)

    sample_text = "Hello stub"
    generated_text = model.run(sample_text)
    assert isinstance(generated_text, GeneratedResult)


def test_bootstrap_and_run_seq2seq():
    """Check if we can bootstrap and run seq2seq models"""

    model = TextGeneration.bootstrap(SEQ2SEQ_LM_MODEL)

    sample_text = "Hello stub"
    generated_text = model.run(sample_text)
    assert isinstance(generated_text, GeneratedResult)


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
        new_model = TextGeneration.load(model_dir)
        sample_text = "Hello stub"
        generated_text = new_model.run(sample_text)
        assert isinstance(generated_text, GeneratedResult)
