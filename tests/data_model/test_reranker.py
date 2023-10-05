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
"""Test for reranker
"""

# Local
from caikit_nlp import data_model as dm

from typing import Dict
import random
import string

## Setup #########################################################################

input_document = {
    "text": "this is the input text",
    "title": "some title attribute here",
    "anything": "another string attribute",
    "_text": "alternate _text here",
}

# TODO: don't use random (could flake).  Make sure to use a working document.
key = ''.join(random.choices(string.ascii_letters, k=20))
value = ''.join(random.choices(string.printable, k=100))
print(key)
print(value)
input_random_document = {
    key: value
}

input_documents = [input_document, input_random_document]
input_queries = []

## Tests ########################################################################


def _compare_rerank_document_and_dict(rerank_doc: dm.RerankDocument, d: Dict):
    assert isinstance(rerank_doc, dm.RerankDocument)
    assert isinstance(rerank_doc.document, Dict)
    assert isinstance(rerank_doc.document["text"], str)
    assert rerank_doc.document["text"] == d["text"]
    assert rerank_doc.document == d


def test_rerank_document():
    new_dm_from_init = dm.RerankDocument(document=input_document)

    _compare_rerank_document_and_dict(new_dm_from_init, input_document)

    # Test proto
    proto_from_dm = new_dm_from_init.to_proto()
    new_dm_from_proto = dm.RerankDocument.from_proto(proto_from_dm)
    _compare_rerank_document_and_dict(new_dm_from_proto, input_document)

    # Test json
    json_from_dm = new_dm_from_init.to_json()
    new_dm_from_json = dm.RerankDocument.from_json(json_from_dm)
    _compare_rerank_document_and_dict(new_dm_from_json, input_document)


def test_rerank_documents():
    in_docs = [dm.RerankDocument(doc) for doc in input_documents]
    new_dm_from_init = dm.RerankDocuments(documents=in_docs)
    assert isinstance(new_dm_from_init, dm.RerankDocuments)
    out_docs = [d.document for d in new_dm_from_init.documents]
    assert out_docs == input_documents

    # Test proto
    proto_from_dm = new_dm_from_init.to_proto()
    new_dm_from_proto = dm.RerankDocuments.from_proto(proto_from_dm)
    out_docs = [d["document"] for d in new_dm_from_proto.documents]
    assert out_docs == input_documents

    # Test json
    json_from_dm = new_dm_from_init.to_json()
    new_dm_from_json = dm.RerankDocuments.from_json(json_from_dm)
    out_docs = [d["document"] for d in new_dm_from_json.documents]
    assert out_docs == input_documents
