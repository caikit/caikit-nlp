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

dummy_generated_response = dm.GeneratedResult(
    generated_token_count=2, text="foo bar", stop_reason=dm.StopReason.TIME_LIMIT
)

## Tests ########################################################################


def test_all_fields_accessible():
    generated_response = dm.GeneratedResult(
        generated_token_count=2,
        text="foo bar",
        stop_reason=dm.StopReason.STOP_SEQUENCE,
    )
    # assert all((hasattr(obj, field) for field in obj.fields))
    assert generated_response.generated_token_count == 2
    assert generated_response.text == "foo bar"
    assert generated_response.stop_reason == dm.StopReason.STOP_SEQUENCE


def test_from_proto_and_back():
    new = dm.GeneratedResult.from_proto(dummy_generated_response.to_proto())
    assert new.generated_token_count == 2
    assert new.text == "foo bar"
    assert new.stop_reason == dm.StopReason.TIME_LIMIT.value


def test_from_json_and_back():
    new = dm.GeneratedResult.from_json(dummy_generated_response.to_json())
    assert new.generated_token_count == 2
    assert new.text == "foo bar"
    assert new.stop_reason == dm.StopReason.TIME_LIMIT.value
