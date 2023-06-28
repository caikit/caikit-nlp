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
"""These interfaces can be promoted to caikit/caikit for wider usage
when applicable to multiple modules
"""
# Standard
from typing import List

# First Party
from caikit.core import DataObjectBase
import caikit


@caikit.core.dataobject(package="caikit_data_model.caikit_nlp")
class Classification(DataObjectBase):
    label: str
    score: float


@caikit.core.dataobject(package="caikit_data_model.caikit_nlp")
class ClassificationResult(DataObjectBase):
    results: List[Classification]


# NOTE: This is meant to align with the HuggingFace token classification task:
# https://huggingface.co/docs/transformers/tasks/token_classification#inference
# The field `word` does not necessarily correspond to a single "word",
# and `entity` may not always be applicable beyond "entity" in the NER
# (named entity recognition) sense
@caikit.core.dataobject(package="caikit_data_model.caikit_nlp")
class TokenClassification(DataObjectBase):
    start: int
    end: int
    word: str  # could be thought of as text
    entity: str  # could be thought of as label
    entity_group: str  # could be thought of as aggregate label, if applicable
    score: float


@caikit.core.dataobject(package="caikit_data_model.caikit_nlp")
class TokenClassificationResult(DataObjectBase):
    results: List[TokenClassification]
