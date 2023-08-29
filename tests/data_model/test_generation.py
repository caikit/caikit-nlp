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
from caikit_nlp.data_model import ExponentialDecayLengthPenalty

## Setup #########################################################################

dummy_exponential_decay_length_penalty = ExponentialDecayLengthPenalty(
    start_index=1, decay_factor=0.95
)

## Tests ########################################################################

### Exponential Decay Length Penalty
def test_exponential_decay_length_penalty_all_fields_accessible():
    assert dummy_exponential_decay_length_penalty.start_index == 1
    assert dummy_exponential_decay_length_penalty.decay_factor == 0.95


def test_sampling_parameters_from_proto_and_back():
    new = ExponentialDecayLengthPenalty.from_proto(
        dummy_exponential_decay_length_penalty.to_proto()
    )
    assert new.start_index == 1
    assert new.decay_factor == 0.95


def test_sampling_parameters_from_json_and_back():
    new = ExponentialDecayLengthPenalty.from_json(
        dummy_exponential_decay_length_penalty.to_json()
    )
    assert new.start_index == 1
    assert new.decay_factor == 0.95
