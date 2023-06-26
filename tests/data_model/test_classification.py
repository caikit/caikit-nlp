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

classification1 = dm.Classification(label="temperature", score=0.71)

classification2 = dm.Classification(label="conditions", score=0.98)

classification_result = dm.ClassificationResult(
    results=[classification1, classification2]
)

token_classification1 = dm.TokenClassification(
    start=0, end=5, word="moose", entity="animal", score=0.8
)
token_classification2 = dm.TokenClassification(
    start=7, end=12, word="goose", entity="animal", score=0.7
)
token_classification_result = dm.TokenClassificationResult(
    results=[token_classification1, token_classification2]
)

## Tests ########################################################################

### Classification


def test_classification_all_fields_accessible():
    classification_result = dm.Classification(label="temperature", score=0.71)
    assert classification_result.label == "temperature"
    assert classification_result.score == 0.71


def test_classification_from_proto_and_back():
    new = dm.Classification.from_proto(classification1.to_proto())
    assert new.label == "temperature"
    assert new.score == 0.71


def test_classification_from_json_and_back():
    new = dm.Classification.from_json(classification1.to_json())
    assert new.label == "temperature"
    assert new.score == 0.71


### ClassificationResult


def test_classification_result_all_fields_accessible():
    classification_result = dm.ClassificationResult(results=[classification1])
    assert classification_result.results[0].label == "temperature"
    assert classification_result.results[0].score == 0.71


def test_classification_result_from_proto_and_back():
    new = dm.ClassificationResult.from_proto(classification_result.to_proto())
    assert new.results[0].label == "temperature"
    assert new.results[0].score == 0.71
    assert new.results[1].label == "conditions"
    assert new.results[1].score == 0.98


def test_classification_result_from_json_and_back():
    new = dm.ClassificationResult.from_json(classification_result.to_json())
    assert new.results[0].label == "temperature"
    assert new.results[0].score == 0.71
    assert new.results[1].label == "conditions"
    assert new.results[1].score == 0.98


### TokenClassification


def test_token_classification_all_fields_accessible():
    token_classification = dm.TokenClassification(
        start=0,
        end=28,
        word="The cow jumped over the moon",
        entity="neutral",
        score=0.6,
    )
    assert token_classification.start == 0
    assert token_classification.end == 28
    assert token_classification.word == "The cow jumped over the moon"
    assert token_classification.entity == "neutral"
    assert token_classification.score == 0.6


def test_classification_from_proto_and_back():
    new = dm.TokenClassification.from_proto(token_classification1.to_proto())
    assert new.start == 0
    assert new.word == "moose"
    assert new.score == 0.8


def test_classification_from_json_and_back():
    new = dm.TokenClassification.from_json(token_classification1.to_json())
    assert new.start == 0
    assert new.word == "moose"
    assert new.score == 0.8


### TokenClassificationResult


def test_token_classification_result_all_fields_accessible():
    token_classification_result = dm.TokenClassificationResult(
        results=[token_classification1]
    )
    assert token_classification_result.results[0].start == 0
    assert token_classification_result.results[0].word == "moose"
    assert token_classification_result.results[0].score == 0.8


def test_token_classification_result_from_proto_and_back():
    new = dm.TokenClassificationResult.from_proto(
        token_classification_result.to_proto()
    )
    assert new.results[0].start == 0
    assert new.results[0].word == "moose"
    assert new.results[0].score == 0.8
    assert new.results[1].end == 12
    assert new.results[1].entity == "animal"


def test_classification_result_from_json_and_back():
    new = dm.TokenClassificationResult.from_json(token_classification_result.to_json())
    assert new.results[0].start == 0
    assert new.results[0].word == "moose"
    assert new.results[0].score == 0.8
    assert new.results[1].end == 12
    assert new.results[1].entity == "animal"
