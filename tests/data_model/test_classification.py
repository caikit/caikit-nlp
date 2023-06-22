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

dummy_classification_prediction = dm.ClassificationPrediction(classifications=[dummy_classification1, dummy_classification2])

## Tests ########################################################################

### Classification

def test_classification_all_fields_accessible():
    classification_result = dm.Classification(
        label="temperature", score=0.71
    )
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


### ClassificationPrediction

def test_classification_prediction_all_fields_accessible():
    classification_prediction_result = dm.ClassificationPrediction(
        classifications=[dummy_classification1]
    )
    assert classification_prediction_result.classifications[0].label == "temperature"
    assert classification_prediction_result.classifications[0].score == 0.71


def test_classification_prediction_from_proto_and_back():
    new = dm.ClassificationPrediction.from_proto(dummy_classification_prediction.to_proto())
    assert new.classifications[0].label == "temperature"
    assert new.classifications[0].score == 0.71
    assert new.classifications[1].label == "conditions"
    assert new.classifications[1].score == 0.98


def test_classification_prediction_from_json_and_back():
    new = dm.ClassificationPrediction.from_json(dummy_classification_prediction.to_json())
    assert new.classifications[0].label == "temperature"
    assert new.classifications[0].score == 0.71
    assert new.classifications[1].label == "conditions"
    assert new.classifications[1].score == 0.98
