"""Tests for data type related utils, e.g., for interacting with serialized torch types.
"""
# Third Party
import pytest
import torch

# Local
from caikit_nlp.toolkit.data_type_utils import get_torch_dtype, str_to_torch_dtype

### Tests for converting from strings / types / None -> torch data types


def test_get_torch_dtype_from_str():
    """Ensure that we can parse a data type from a string."""
    assert torch.float32 is get_torch_dtype("float32")


def test_get_torch_dtype_from_dtype():
    """Ensure that if a data type is provided from pytorch, we simply return it."""
    assert torch.float32 is get_torch_dtype(torch.float32)


def test_get_torch_dtype_from_bad_type():
    """Ensure that if a type we can't coerce to a pytorch dtype is given, we get a TypeError."""
    with pytest.raises(TypeError):
        assert torch.float32 is get_torch_dtype(100)


def test_get_torch_dtype_from_bad_str():
    """Ensure that if an invalid type string is given, we get a ValueError."""
    with pytest.raises(ValueError):
        assert torch.float32 is get_torch_dtype("not a valid attr of pytorch")


### Tests for converting from strings -> torch data types
def test_str_to_torch_dtype():
    """Ensure that we can parse a data type from a string."""
    assert str_to_torch_dtype("float32") is torch.float32


def test_str_to_torch_dtype_invalid_attr():
    """Ensure that we raise ValueError if an incorrect type str is provided."""
    with pytest.raises(ValueError):
        str_to_torch_dtype("not a valid attr of pytorch")


def test_str_to_torch_dtype_bad_attr():
    """Ensure that we raise ValueError if a non type property of torch is provided."""
    with pytest.raises(ValueError):
        str_to_torch_dtype("nn")
