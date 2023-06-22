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

# Local
from caikit_nlp import data_model as dm

## Setup #########################################################################

dummy_span = dm.Span(start=0, end=11, text="Hello World")

## Tests ########################################################################

def test_all_fields_accessible():
    span = dm.Span(
        start=0, end=11, text="Hello World"
    )
    assert span.start == 0
    assert span.end == 11
    assert span.text == "Hello World"

def test_from_proto_and_back():
    new = dm.Span.from_proto(dummy_span.to_proto())
    assert new.start == 0
    assert new.end == 11
    assert new.text == "Hello World"


def test_from_json_and_back():
    new = dm.Span.from_json(dummy_span.to_json())
    assert new.start == 0
    assert new.end == 11
    assert new.text == "Hello World"
