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
# Standard
from typing import List

# Third Party
from transformers import AutoModelForSeq2SeqLM
from transformers.models.auto import modeling_auto

# First Party
from caikit.core import resource
from caikit.core.toolkit import error_handler
import alog

# Local
from .base import PretrainedModelBase
from ...data_model import PromptOutputModelType


log = alog.use_channel("HFRBAS")
error = error_handler.get(log)


@resource(
    id="6759e891-287b-405b-bd8b-54a4a4d51c25",
    name="HF Transformers Auto Seq2Seq LM",
    version="0.1.0",
)
class HFAutoSeq2SeqLM(PretrainedModelBase):
    """This resource wraps a handle to a Huggingface
    AutoModelForSeq2SeqLM
    """

    MODEL_TYPE = AutoModelForSeq2SeqLM
    SUPPORTED_MODEL_TYPES = modeling_auto.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
    TASK_TYPE = "SEQ_2_SEQ_LM"
    PROMPT_OUTPUT_TYPES = [PromptOutputModelType.ENCODER]
    MAX_NUM_TRANSFORMERS = 2

    @classmethod
    def get_num_transformers_submodules(
        cls, output_model_types: List[PromptOutputModelType]
    ):
        """Return number of applicable transformer submodules"""
        num_transformer_submodules = 0
        if PromptOutputModelType.ENCODER in output_model_types:
            num_transformer_submodules += 1
        if PromptOutputModelType.DECODER in output_model_types:
            num_transformer_submodules += 1
        error.value_check(
            "<FPT71505742E>", 0 < num_transformer_submodules <= cls.MAX_NUM_TRANSFORMERS
        )
        return num_transformer_submodules
