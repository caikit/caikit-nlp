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
from typing import List, Optional

# First Party
from caikit.core import DataObjectBase, dataobject
from caikit.core.data_model.json_dict import JsonDict


@dataobject()
class RerankScore(DataObjectBase):
    """The score for one document (one query)"""

    document: Optional[JsonDict]
    index: int
    score: float
    text: Optional[str]


@dataobject()
class RerankQueryResult(DataObjectBase):
    """Result for one query in a rerank task"""

    query: Optional[str]
    scores: List[RerankScore]


@dataobject()
class RerankPrediction(DataObjectBase):
    """Result for a rerank task"""

    results: List[RerankQueryResult]
