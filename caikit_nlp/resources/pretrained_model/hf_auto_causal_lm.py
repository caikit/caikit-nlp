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
"""
Huggingface auto causal LM resource type
"""

# Third Party
from transformers import AutoModelForCausalLM
from transformers.models.auto import modeling_auto

# First Party
from caikit.core.modules import module

# Local
from ...data_model import PromptOutputModelType
from .base import PretrainedModelBase


@module(
    id="6759e891-287b-405b-bd8b-54a4a4d51c24",
    name="HF Transformers Auto Causal LM",
    version="0.1.0",
)
class HFAutoCausalLM(PretrainedModelBase):
    """This resource (module) wraps a handle to a Huggingface AutoModelForCausalLM"""

    MODEL_TYPE = AutoModelForCausalLM
    SUPPORTED_MODEL_TYPES = modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    TASK_TYPE = "CAUSAL_LM"
    PROMPT_OUTPUT_TYPES = [PromptOutputModelType.DECODER]
    MAX_NUM_TRANSFORMERS = 1
