# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import Optional, Union

# Third Party
import torch

# First Party
from caikit import get_config
from caikit.core.exceptions import error_handler
import alog

log = alog.use_channel("DATA_UTIL")
error = error_handler.get(log)


def str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    """Given a string representation of a Torch data type, convert it to the actual torch dtype.

    Args:
        dtype_str: String representation of Torch dtype to be used; this should be an attr
        of the torch library whose value is a dtype.

    Returns:
        torch.dtype
            Data type of the Torch class being used.
    """
    dt = getattr(torch, dtype_str, None)
    if not isinstance(dt, torch.dtype):
        error("<NLP85554812E>", ValueError(f"Unrecognized data type: {dtype_str}"))
    return dt


def get_torch_dtype(dtype: Optional[Union[str, torch.dtype]] = None) -> torch.dtype:
    """Get the Torch data type to be used for interacting with a model.

    Args:
        dtype: Optional[Union[str, torch.dtype]]
            If dtype is a torch.dtype, returns it; if it's a string, grab it from the Torch lib.
            If None is provided, fall back to the default type in config, which can be
            overridden via environment variable.

    Returns:
        torch.dtype
            Torch data type to be used.
    """
    error.type_check("<NLP84274822E>", torch.dtype, str, dtype=dtype, allow_none=True)
    # If a Torch dtype is passed, nothing to do
    if isinstance(dtype, torch.dtype):
        return dtype
    # If None/empty str was provided, fall back to config / env var override
    if not dtype:
        return str_to_torch_dtype(get_config().torch_dtype)
    # Otherwise convert it from a string
    return str_to_torch_dtype(dtype)
