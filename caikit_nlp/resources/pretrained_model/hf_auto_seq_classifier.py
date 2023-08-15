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
Huggingface auto sequence classifier resource type
"""
# Standard
from typing import Callable, Tuple

# Third Party
from transformers import AutoModelForSequenceClassification
from transformers.models.auto import modeling_auto

# First Party
from caikit.core.modules import module

# Local
from .base import PretrainedModelBase


@module(
    id="6759e891-287b-405b-bd8b-54a4a4d51c23",
    name="HF Transformers Auto Sequence Classifier",
    version="0.1.0",
)
class HFAutoSequenceClassifier(PretrainedModelBase):
    """This resource (module) wraps a handle to a huggingface
    AutoModelForSequenceClassification
    """

    MODEL_TYPE = AutoModelForSequenceClassification
    SUPPORTED_MODEL_TYPES = (
        modeling_auto.MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
    )
    TASK_TYPE = "SEQ_CLS"
    PROMPT_OUTPUT_TYPES = []
    MAX_NUM_TRANSFORMERS = 1

    @classmethod
    def bootstrap(cls, *args, **kwargs) -> "HFAutoSequenceClassifier":
        """Bootstrap from a huggingface model

        See help(PretrainedModelBase)
        """
        return super().bootstrap(*args, return_dict=True, **kwargs)

    @staticmethod
    def tokenize_function(*args, **kwargs) -> Tuple[Callable, bool]:
        raise NotImplementedError(
            "Tokenize func not implemented for sequence classifier"
        )
