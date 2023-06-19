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
from typing import List

# First Party
from caikit.core import DataObjectBase
import caikit


# Could eventually refactor to inherit from Span, Classification
# with caikit >= 0.8.0 upgrade
@caikit.core.dataobject(package="caikit_data_model.caikit_nlp")
class SpanClassification(DataObjectBase):
    start: int
    end: int
    text: str  # maps to word in HF NER, could replace
    label: str  # more generic than entity_group, could replace
    score: float


@caikit.core.dataobject(package="caikit_data_model.caikit_nlp")
class TextContentResults(DataObjectBase):
    results: List[SpanClassification]
