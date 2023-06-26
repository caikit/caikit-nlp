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

dummy_classification1 = dm.Classification(label="temperature", score=0.71)

dummy_classification2 = dm.Classification(label="conditions", score=0.98)

dummy_classification_result = dm.ClassificationResult(
    results=[dummy_classification1, dummy_classification2]
)

## Tests ########################################################################

### Classification


def test_classification_all_fields_accessible():
    classification_result = dm.Classification(label="temperature", score=0.71)
    assert classification_result.label == "temperature"
    assert classification_result.score == 0.71


def test_classification_from_proto_and_back():
    new = dm.Classification.from_proto(dummy_classification1.to_proto())
    assert new.label == "temperature"
    assert new.score == 0.71


def test_classification_from_json_and_back():
    new = dm.Classification.from_json(dummy_classification1.to_json())
    assert new.label == "temperature"
    assert new.score == 0.71


### ClassificationResult


def test_classification_result_all_fields_accessible():
    classification_result = dm.ClassificationResult(results=[dummy_classification1])
    assert classification_result.results[0].label == "temperature"
    assert classification_result.results[0].score == 0.71


def test_classification_result_from_proto_and_back():
    new = dm.ClassificationResult.from_proto(dummy_classification_result.to_proto())
    assert new.results[0].label == "temperature"
    assert new.results[0].score == 0.71
    assert new.results[1].label == "conditions"
    assert new.results[1].score == 0.98


def test_classification_result_from_json_and_back():
    new = dm.ClassificationResult.from_json(dummy_classification_result.to_json())
    assert new.results[0].label == "temperature"
    assert new.results[0].score == 0.71
    assert new.results[1].label == "conditions"
    assert new.results[1].score == 0.98
