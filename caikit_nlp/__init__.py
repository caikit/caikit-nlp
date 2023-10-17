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
"""Caikit prompt tuning library
"""
# Standard
import os

# First Party
from caikit.core.model_manager import *

# Import the model management semantics from the core
import caikit

# Local
# Import subpackages
from . import config, data_model, model_management
from .config import *
from .data_model import *
from .modules import *
from .resources import *
from .version import __version__, __version_tuple__

# Configure the library with library-specific configuration file
CONFIG_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "config", "config.yml")
)

caikit.configure(CONFIG_PATH)
