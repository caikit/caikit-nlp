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

# Standard
import random
import string

# Local
from caikit_nlp import data_model as dm

## Setup #########################################################################

input_document = {
    "text": "this is the input text",
    "_text": "alternate _text here",
    "title": "some title attribute here",
    "anything": "another string attribute",
    "str_test": "test string",
    "int_test": 1234,
    "float_test": 9876.4321,
}

# TODO: don't use random (could flake).  Make sure to use a working document.
key = "".join(random.choices(string.ascii_letters, k=20))
value = "".join(random.choices(string.printable, k=100))
input_random_document = {
    "text": "".join(random.choices(string.printable, k=100)),
    "random_str": "".join(random.choices(string.printable, k=100)),
    "random_int": random.randint(-99999, 99999),
    "random_float": random.uniform(-99999, 99999),
}

input_documents = [input_document, input_random_document]
input_queries = []

## Tests ########################################################################


def test_rerank_documents():
    in_docs = input_documents
    new_dm_from_init = dm.RerankDocuments(documents=in_docs)
    assert isinstance(new_dm_from_init, dm.RerankDocuments)
    assert new_dm_from_init.documents == input_documents

    # Test proto
    proto_from_dm = new_dm_from_init.to_proto()
    new_dm_from_proto = dm.RerankDocuments.from_proto(proto_from_dm)
    assert new_dm_from_proto.documents == input_documents

    # Test json
    json_from_dm = new_dm_from_init.to_json()
    new_dm_from_json = dm.RerankDocuments.from_json(json_from_dm)
    assert new_dm_from_json.documents == input_documents
