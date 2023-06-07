"""Tests for verbalizer substitution utils.
"""
# Third Party
import pytest

# Local
from caikit_nlp.toolkit.verbalizer_utils import is_valid_verbalizer, render_verbalizer
import caikit_nlp

SAMPLE_DM = caikit_nlp.data_model.GenerationTrainRecord(
    input="my input text", output="my output text"
)
SAMPLE_DICT = SAMPLE_DM.to_dict()

### Happy path rendering cases
def test_is_valid_verbalizer():
    """Ensure that when we have happy verbalizers, it's easy to check."""
    assert is_valid_verbalizer("{{input}}") == True
    assert is_valid_verbalizer("Input: {{input}}") == True
    assert is_valid_verbalizer("Input: {{input}} Output: {{output}}") == True


def test_raw_verbalizer_is_preserved():
    """Ensure that if no placeholders are provided, the raw string is returned."""
    # NOTE: you would never want to do this because it means your input string is hardcoded!
    # However, we the responsibility of checking this up to the verbalizer validator so that
    # we can validate up front, rather than checking lazily at train time etc.
    verbalizer_template = "text: foo label: bar"
    expected_render_result = verbalizer_template
    # Check that we get the same behavior from both the DM / Dict based verbalizer replacement
    assert render_verbalizer(verbalizer_template, SAMPLE_DM) == expected_render_result
    assert render_verbalizer(verbalizer_template, SAMPLE_DICT) == expected_render_result


def test_verbalizer_substitution():
    """Ensure that if placeholders are used, class attrs are rendered into placeholders."""
    verbalizer_template = "text: {{input}} label: {{output}}"
    expected_render_result = "text: {} label: {}".format(
        SAMPLE_DM.input, SAMPLE_DM.output
    )
    assert render_verbalizer(verbalizer_template, SAMPLE_DM) == expected_render_result
    assert render_verbalizer(verbalizer_template, SAMPLE_DICT) == expected_render_result


def test_invalid_value_verbalizer_substitution():
    """Ensure that if we try to render a placeholder that is a bad property name, we skip it."""
    verbalizer_template = "text: {{foo bar}}"
    expected_render_result = verbalizer_template
    assert render_verbalizer(verbalizer_template, SAMPLE_DM) == expected_render_result
    assert render_verbalizer(verbalizer_template, SAMPLE_DICT) == expected_render_result


def test_empty_verbalizer_substitution():
    """Ensure that if we try to render an empty placeholder, we skip it."""
    verbalizer_template = "text: {{}} label: {{}}"
    expected_render_result = verbalizer_template
    assert render_verbalizer(verbalizer_template, SAMPLE_DM) == expected_render_result
    assert render_verbalizer(verbalizer_template, SAMPLE_DICT) == expected_render_result


### sad path rendering cases
def test_is_invalid_verbalizer():
    """Ensure that when we have happy verbalizers, it's easy to check."""
    assert is_valid_verbalizer(100) == False
    assert is_valid_verbalizer("") == False
    assert is_valid_verbalizer("source") == False
    assert is_valid_verbalizer("{{this is not a valid placeholder}}") == False


def test_invalid_attribute_verbalizer_substitution():
    """Ensure that if we try to render a placeholder that doesn't exist on our DM, we fail."""
    verbalizer_template = "text: {{sadness}}"
    with pytest.raises(AttributeError):
        render_verbalizer(verbalizer_template, SAMPLE_DM)


def test_invalid_key_verbalizer_substitution():
    """Ensure that if we try to render a placeholder that doesn't exist on our dict, we fail."""
    verbalizer_template = "text: {{sadness}}"
    with pytest.raises(KeyError):
        render_verbalizer(verbalizer_template, SAMPLE_DICT)
