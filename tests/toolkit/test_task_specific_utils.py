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

# Third Party
import pytest

# First Party
from caikit.interfaces.nlp.data_model import ClassificationTrainRecord

# Local
from caikit_nlp.data_model import GenerationTrainRecord
from caikit_nlp.toolkit.task_specific_utils import convert_to_generation_record


def test_convert_classification_train_record_to_generation_record():
    classification_train_record = ClassificationTrainRecord(
        text="foo bar", labels=["label1"]
    )
    generated_train = convert_to_generation_record(classification_train_record)
    assert isinstance(generated_train, GenerationTrainRecord)
    assert generated_train.input == "foo bar"
    assert generated_train.output == "label1"


def test_convert_generation_record_to_generation_record():
    generation_train_record = GenerationTrainRecord(input="foo bar", output="label1")
    generated_train = convert_to_generation_record(generation_train_record)
    assert isinstance(generated_train, GenerationTrainRecord)
    assert generated_train.input == generation_train_record.input
    assert generated_train.output == generation_train_record.output


# When we support integer labels
# def test_convert_classification_train_record_to_generation_record_numeric_labels():
#     classification_train_record = dm.ClassificationTrainRecord(
#         text="foo bar", labels=[1]
#     )
#     generated_train = convert_to_generation_record(classification_train_record)
#     assert isinstance(generated_train, dm.GenerationTrainRecord)
#     assert generated_train.input == classification_train_record.text
#     assert generated_train.output == "1"


def test_convert_to_generation_record_gives_error_with_unsupported_type():
    string_record = "test record"
    with pytest.raises(TypeError):
        convert_to_generation_record(string_record)
