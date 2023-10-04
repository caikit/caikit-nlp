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
"""Test for embedding vectors
"""
# Third Party
import numpy as np

# Local
from caikit_nlp import data_model as dm

## Setup #########################################################################

RANDOM_SEED = 77
DUMMY_VECTOR_SHAPE = (5,)

np.random.seed(RANDOM_SEED)

random_number_generator = np.random.default_rng()

random_numpy_vector1d_float32 = random_number_generator.random(
    DUMMY_VECTOR_SHAPE, dtype=np.float32
)
random_numpy_vector1d_float64 = random_number_generator.random(
    DUMMY_VECTOR_SHAPE, dtype=np.float64
)
random_python_vector1d_float = random_numpy_vector1d_float32.tolist()

vector1d_response_float32 = dm.Vector1D(data=[random_python_vector1d_float])

## Tests ########################################################################


def test_vector1d_dm_float():
    dm_result = dm.Vector1D(data=dm.PyFloatSequence(random_python_vector1d_float))

    assert isinstance(dm_result.data.values[0], float)
    assert dm_result.data.values == random_python_vector1d_float

    # Test proto
    vector_in_proto = dm_result.to_proto()
    new_dm_from_proto = dm.Vector1D.from_proto(vector_in_proto)
    assert isinstance(new_dm_from_proto.data.values[0], float)
    assert new_dm_from_proto.data.values == random_python_vector1d_float

    # Test json
    vector_in_json = dm_result.to_json()
    new_dm_from_json = dm.Vector1D.from_json(vector_in_json)
    assert isinstance(new_dm_from_json.data.values[0], float)

    assert new_dm_from_json.data.values == random_python_vector1d_float


def test_vector1d_dm_numpy_float32():
    dm_result = dm.Vector1D(data=dm.NpFloat32Sequence(random_numpy_vector1d_float32))

    assert isinstance(dm_result.data.values[0], np.float32)
    print(type(dm_result.data.values[0]))
    np.testing.assert_array_equal(dm_result.data.values, random_numpy_vector1d_float32)

    # Test proto
    vector_in_proto = dm_result.to_proto()
    assert isinstance(vector_in_proto.data_npfloat32sequence.values[0], float)
    new_dm_from_proto = dm.Vector1D.from_proto(vector_in_proto)
    one_of = new_dm_from_proto.which_oneof("data")
    d = getattr(new_dm_from_proto, one_of)
    assert isinstance(new_dm_from_proto.data_npfloat32sequence.values[0], float)
    assert isinstance(new_dm_from_proto.data.values[0], float)  # Back to np???? TODO:

    # check if proto does contain value in float32
    assert isinstance(vector_in_proto.data_npfloat32sequence.values[0], float)
    np.testing.assert_array_equal(
        new_dm_from_proto.data.values, random_numpy_vector1d_float32
    )
    assert (
        vector_in_proto.data_npfloat32sequence.values[0]
        == random_numpy_vector1d_float32[0]
    )

    # Test json
    vector_in_json = dm_result.to_json()
    new_dm_from_json = dm.Vector1D.from_json(vector_in_json)
    # NOTE: When converting from json, the resultant will not be of same type since
    # json cannot store numpy data type
    np.testing.assert_array_equal(
        new_dm_from_json.data.values, random_numpy_vector1d_float32
    )


def test_vector1d_dm_numpy_float64():
    dm_result = dm.Vector1D(data=dm.NpFloat64Sequence(random_numpy_vector1d_float64))

    np.testing.assert_array_equal(dm_result.data.values, random_numpy_vector1d_float64)

    # Test proto
    vector_in_proto = dm_result.to_proto()
    # check if proto does contain value in float64
    assert (
        vector_in_proto.data_npfloat64sequence.values[0]
        == random_numpy_vector1d_float64[0]
    )
    new_dm_from_proto = dm.Vector1D.from_proto(vector_in_proto)
    np.testing.assert_array_equal(
        new_dm_from_proto.data.values, random_numpy_vector1d_float64
    )

    # NOTE: Since we do not have a way of supporting float16 type etc currently, once the data is converted to proto / json
    # it will become python list only.
    np.testing.assert_array_equal(
        new_dm_from_proto.data.values, random_numpy_vector1d_float64
    )
    assert (
        vector_in_proto.data_npfloat64sequence.values[0]
        == random_numpy_vector1d_float64[0]
    )

    # Test json
    vector_in_json = dm_result.to_json()
    new_dm_from_json = dm.Vector1D.from_json(vector_in_json)
    np.testing.assert_array_equal(
        new_dm_from_json.data.values, random_numpy_vector1d_float64
    )


def test_embedding_result_dm_float64_1d():
    dm_result = dm.EmbeddingResult(
        data=[dm.Vector1D(data=dm.NpFloat64Sequence(random_numpy_vector1d_float64))]
    )

    # Test proto
    vector_in_proto = dm_result.to_proto()
    new_dm_from_proto = dm.EmbeddingResult.from_proto(vector_in_proto)
    assert isinstance(new_dm_from_proto.data, list)
    assert isinstance(new_dm_from_proto.data[0], dm.Vector1D)
    np.testing.assert_array_equal(
        new_dm_from_proto.data[0].data.values, random_numpy_vector1d_float64
    )

    # Test json
    vector_in_json = dm_result.to_json()
    new_dm_from_json = dm.EmbeddingResult.from_json(vector_in_json)
    assert isinstance(new_dm_from_json.data, list)
    assert isinstance(new_dm_from_json.data[0], dm.Vector1D)
    np.testing.assert_array_equal(
        new_dm_from_json.data[0].data.values, random_numpy_vector1d_float64
    )


def test_embedding_result_dm_float64_2d():
    dm_result = dm.EmbeddingResult(
        data=[
            dm.Vector1D(data=dm.NpFloat64Sequence(random_numpy_vector1d_float64)),
            dm.Vector1D(data=dm.NpFloat32Sequence(random_numpy_vector1d_float32)),
            dm.Vector1D(data=dm.PyFloatSequence(random_python_vector1d_float)),
        ]
    )

    # Test proto
    vector_in_proto = dm_result.to_proto()
    new_dm_from_proto = dm.EmbeddingResult.from_proto(vector_in_proto)
    assert isinstance(new_dm_from_proto.data, list)
    assert len(new_dm_from_proto.data) == 3
    assert isinstance(new_dm_from_proto.data[0], dm.Vector1D)
    np.testing.assert_array_equal(
        new_dm_from_proto.data[0].data.values, random_numpy_vector1d_float64
    )
    np.testing.assert_array_equal(
        new_dm_from_proto.data[1].data.values, random_numpy_vector1d_float32
    )
    np.testing.assert_array_equal(
        new_dm_from_proto.data[2].data.values, random_python_vector1d_float
    )

    # Test json
    vector_in_json = dm_result.to_json()
    new_dm_from_json = dm.EmbeddingResult.from_json(vector_in_json)
    assert isinstance(new_dm_from_json.data, list)
    assert isinstance(new_dm_from_json.data[0], dm.Vector1D)
    assert len(new_dm_from_proto.data) == 3
    np.testing.assert_array_equal(
        new_dm_from_json.data[0].data.values, random_numpy_vector1d_float64
    )
