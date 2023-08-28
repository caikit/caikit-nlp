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
from caikit_nlp.data_model import (
    DecodingParameters,
    SamplingParameters,
    StoppingCriteria,
)

## Setup #########################################################################

dummy_exponential_decay_length_penalty = (
    DecodingParameters.ExponentialDecayLengthPenalty(start_index=1, decay_factor=0.95)
)
dummy_sampling_parameters = DecodingParameters(
    repetition_penalty=1.2,
    exponential_decay_length_penalty=dummy_exponential_decay_length_penalty,
)

dummy_sampling_parameters = SamplingParameters(
    temperature=0.5, top_k=0, top_p=0, typical_p=0.2, seed=42
)

dummy_stopping_criteria = StoppingCriteria(
    max_new_tokens=200, min_new_tokens=50, time_limit_millis=0, stop_sequences=["Test"]
)

## Tests ########################################################################

### Decoding Parameters
def test_sampling_parameters_all_fields_accessible():
    assert dummy_sampling_parameters.repetition_penalty == 1.2
    assert dummy_sampling_parameters.exponential_decay_length_penalty.start_index == 1
    assert (
        dummy_sampling_parameters.exponential_decay_length_penalty.decay_factor == 0.95
    )


def test_sampling_parameters_from_proto_and_back():
    new = DecodingParameters.from_proto(dummy_sampling_parameters.to_proto())
    assert new.repetition_penalty == 1.2
    assert new.exponential_decay_length_penalty.start_index == 1
    assert new.exponential_decay_length_penalty.decay_factor == 0.95


def test_sampling_parameters_from_json_and_back():
    new = DecodingParameters.from_json(dummy_sampling_parameters.to_json())
    assert new.repetition_penalty == 1.2
    assert new.exponential_decay_length_penalty.start_index == 1
    assert new.exponential_decay_length_penalty.decay_factor == 0.95


### Sampling Parameters
def test_sampling_parameters_all_fields_accessible():
    assert dummy_sampling_parameters.temperature == 0.5
    assert dummy_sampling_parameters.top_k == 0
    assert dummy_sampling_parameters.top_p == 0
    assert dummy_sampling_parameters.typical_p == 0.2
    assert dummy_sampling_parameters.seed == 42


def test_sampling_parameters_from_proto_and_back():
    new = SamplingParameters.from_proto(dummy_sampling_parameters.to_proto())
    assert new.temperature == 0.5
    assert new.top_k == 0
    assert new.top_p == 0
    assert new.typical_p == 0.2
    assert new.seed == 42


def test_sampling_parameters_from_json_and_back():
    new = SamplingParameters.from_json(dummy_sampling_parameters.to_json())
    assert new.temperature == 0.5
    assert new.top_k == 0
    assert new.top_p == 0
    assert new.typical_p == 0.2
    assert new.seed == 42


### Stopping Criteria
def test_stopping_criteria_all_fields_accessible():
    assert dummy_stopping_criteria.max_new_tokens == 200
    assert dummy_stopping_criteria.min_new_tokens == 50
    assert dummy_stopping_criteria.time_limit_millis == 0
    assert dummy_stopping_criteria.stop_sequences == ["Test"]


def test_stopping_criteria_from_proto_and_back():
    new = StoppingCriteria.from_proto(dummy_stopping_criteria.to_proto())
    assert new.max_new_tokens == 200
    assert new.min_new_tokens == 50
    assert new.time_limit_millis == 0
    assert new.stop_sequences == ["Test"]


def test_stopping_criteria_from_json_and_back():
    new = StoppingCriteria.from_json(dummy_stopping_criteria.to_json())
    assert new.max_new_tokens == 200
    assert new.min_new_tokens == 50
    assert new.time_limit_millis == 0
    assert new.stop_sequences == ["Test"]
