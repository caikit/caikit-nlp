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

from caikit_nlp import data_model as dm

## Setup #########################################################################

dummy_classification_train_record = dm.ClassificationTrainRecord(
    text="It is 20 degrees today", labels=["temperature"]
)

## Tests ########################################################################


def test_all_fields_accessible():
    dummy_classification_train_record = dm.ClassificationTrainRecord(
        text="It is 20 degrees today", labels=["temperature"]
    )
    assert dummy_classification_train_record.text == "It is 20 degrees today"
    assert dummy_classification_train_record.labels == ["temperature"]

def test_from_proto_and_back():
    new = dm.ClassificationTrainRecord.from_proto(dummy_classification_train_record.to_proto())
    assert new.text == "It is 20 degrees today"
    assert new.labels == ["temperature"]


def test_from_json_and_back():
    new = dm.ClassificationTrainRecord.from_json(dummy_classification_train_record.to_json())
    assert new.text == "It is 20 degrees today"
    assert new.labels == ["temperature"]