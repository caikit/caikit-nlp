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

from caikit.core.data_model.json_dict import JsonDict
from caikit.core import (
    dataobject,
    DataObjectBase,
)

from typing import List, Dict


@dataobject()
class RerankDocument(DataObjectBase):
    """An input document with key of text else _text else empty string used for comparison"""
    document: Dict[str, str]  # TODO: get any JsonDict working for input


@dataobject()
class RerankDocuments(DataObjectBase):
    """An input list of documents"""
    documents: List[RerankDocument]

    @classmethod
    def from_proto(cls, proto):
        return cls([{"document": dict(d.document.items())} for d in proto.documents])


@dataobject()
class RerankScore(DataObjectBase):
    """The score for one document (one query)"""
    document: JsonDict
    corpus_id: int
    score: float


@dataobject()
class RerankQueryResult(DataObjectBase):
    """Result for one query in a rerank task"""
    scores: List[RerankScore]


@dataobject()
class RerankPrediction(DataObjectBase):
    """Result for a rerank task"""
    results: List[RerankQueryResult]
