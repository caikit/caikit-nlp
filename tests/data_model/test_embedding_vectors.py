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
# Standard
from collections import namedtuple

# Third Party
import numpy as np
import pytest

# Local
from caikit_nlp import data_model as dm

## Setup #########################################################################

RANDOM_SEED = 77
DUMMY_VECTOR_SHAPE = (5,)

# To tests the limits of our type-checking, this can replace our legit data objects
TRICK_SEQUENCE = namedtuple("Trick", "values")

np.random.seed(RANDOM_SEED)

random_number_generator = np.random.default_rng()

random_numpy_vector1d_float32 = random_number_generator.random(
    DUMMY_VECTOR_SHAPE, dtype=np.float32
)
random_numpy_vector1d_float64 = random_number_generator.random(
    DUMMY_VECTOR_SHAPE, dtype=np.float64
)
random_python_vector1d_float = random_numpy_vector1d_float32.tolist()

## Tests ########################################################################


@pytest.mark.parametrize(
    "data",
    [dm.Vector1D(dm.PyFloatSequence()), "foo", ["bar"], None],
    ids=type,
)
def test_embedding_result_type_checks(data):
    with pytest.raises(TypeError):
        dm.EmbeddingResult(data=data)


@pytest.mark.parametrize(
    "sequence",
    [
        dm.PyFloatSequence(),
        dm.NpFloat32Sequence(),
        dm.NpFloat64Sequence(),
    ],
    ids=type,
)
def test_empty_sequences(sequence):
    """No type check error with empty sequences"""
    new_dm_from_init = dm.EmbeddingResult(results=[dm.Vector1D(sequence)])
    assert isinstance(new_dm_from_init.results, list)
    for vector in new_dm_from_init.results:
        assert hasattr(vector, "data")
        assert hasattr(vector.data, "values")
        assert vector.data.values is None

    # Test proto
    proto_from_dm = new_dm_from_init.to_proto()
    new_dm_from_proto = dm.EmbeddingResult.from_proto(proto_from_dm)
    assert isinstance(new_dm_from_proto.results, list)
    for vector in new_dm_from_proto.results:
        assert hasattr(vector, "data")
        assert hasattr(vector.data, "values")
        assert vector.data.values is None

    # Test json
    json_from_dm = new_dm_from_init.to_json()
    print("JSON:", json_from_dm)
    new_dm_from_json = dm.EmbeddingResult.from_json(json_from_dm)
    assert isinstance(new_dm_from_json.results, list)
    for vector in new_dm_from_json.results:
        assert hasattr(vector, "data")
        assert hasattr(vector.data, "values")
        assert vector.data.values == []


def test_vector1d_iterator_error():
    """Cannot just shove in an iterator and expect it to work"""
    with pytest.raises(ValueError):
        dm.Vector1D(data=[1.1, 2.2, 3.3])


def test_vector1d_trick():
    """The param check currently allows for objects with values using this trick"""
    dm.Vector1D(data=TRICK_SEQUENCE(values=[1.1, 2.2]))


def _assert_array_check(new_array, data_values, float_type):
    for value in new_array.data.values:
        assert isinstance(value, float_type)
    np.testing.assert_array_equal(new_array.data.values, data_values)


@pytest.mark.parametrize(
    "float_seq_class, random_values, float_type",
    [
        (dm.PyFloatSequence, random_python_vector1d_float, float),
        (dm.NpFloat32Sequence, random_numpy_vector1d_float32, np.float32),
        (dm.NpFloat64Sequence, random_numpy_vector1d_float64, np.float64),
        (TRICK_SEQUENCE, [1.1, 2.2], float),  # Sneaky but tests corner cases for now
    ],
)
def test_vector1d_dm(float_seq_class, random_values, float_type):

    # Test init
    dm_init = dm.Vector1D(data=(float_seq_class(random_values)))
    _assert_array_check(dm_init, random_values, float_type)

    # Test proto
    dm_to_proto = dm_init.to_proto()
    dm_from_proto = dm.Vector1D.from_proto(dm_to_proto)
    _assert_array_check(dm_from_proto, random_values, float_type)

    # Test json
    dm_to_json = dm_init.to_json()
    dm_from_json = dm.Vector1D.from_json(dm_to_json)
    _assert_array_check(
        dm_from_json, random_values, float
    )  # NOTE: always float after json


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
    np.testing.assert_array_equal(dm_result.data.values, random_numpy_vector1d_float32)

    # Test proto
    vector_in_proto = dm_result.to_proto()
    assert isinstance(vector_in_proto.data_npfloat32sequence.values[0], float)  # py

    new_dm_from_proto = dm.Vector1D.from_proto(vector_in_proto)
    assert isinstance(
        new_dm_from_proto.data_npfloat32sequence.values[0], np.float32
    )  # np
    assert isinstance(new_dm_from_proto.data.values[0], np.float32)

    # check if proto does contain value in float32
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
    assert isinstance(new_dm_from_proto.data_npfloat64sequence.values[0], np.float64)
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
        results=[dm.Vector1D(data=dm.NpFloat64Sequence(random_numpy_vector1d_float64))]
    )

    # Test proto
    vector_in_proto = dm_result.to_proto()
    new_dm_from_proto = dm.EmbeddingResult.from_proto(vector_in_proto)
    assert isinstance(new_dm_from_proto.results, list)
    assert isinstance(new_dm_from_proto.results[0], dm.Vector1D)
    np.testing.assert_array_equal(
        new_dm_from_proto.results[0].data.values, random_numpy_vector1d_float64
    )

    # Test json
    vector_in_json = dm_result.to_json()
    new_dm_from_json = dm.EmbeddingResult.from_json(vector_in_json)
    assert isinstance(new_dm_from_json.results, list)
    assert isinstance(new_dm_from_json.results[0], dm.Vector1D)
    np.testing.assert_array_equal(
        new_dm_from_json.results[0].data.values, random_numpy_vector1d_float64
    )


def test_embedding_result_dm_2d():
    dm_result = dm.EmbeddingResult(
        results=[
            dm.Vector1D(data=dm.NpFloat64Sequence(random_numpy_vector1d_float64)),
            dm.Vector1D(data=dm.NpFloat32Sequence(random_numpy_vector1d_float32)),
            dm.Vector1D(data=dm.PyFloatSequence(random_python_vector1d_float)),
        ]
    )

    # Test proto
    vector_in_proto = dm_result.to_proto()
    new_dm_from_proto = dm.EmbeddingResult.from_proto(vector_in_proto)
    assert isinstance(new_dm_from_proto.results, list)
    assert len(new_dm_from_proto.results) == 3
    assert isinstance(new_dm_from_proto.results[0], dm.Vector1D)
    np.testing.assert_array_equal(
        new_dm_from_proto.results[0].data.values, random_numpy_vector1d_float64
    )
    np.testing.assert_array_equal(
        new_dm_from_proto.results[1].data.values, random_numpy_vector1d_float32
    )
    np.testing.assert_array_equal(
        new_dm_from_proto.results[2].data.values, random_python_vector1d_float
    )

    # Test json
    vector_in_json = dm_result.to_json()
    new_dm_from_json = dm.EmbeddingResult.from_json(vector_in_json)
    assert isinstance(new_dm_from_json.results, list)
    assert isinstance(new_dm_from_json.results[0], dm.Vector1D)
    assert len(new_dm_from_proto.results) == 3
    np.testing.assert_array_equal(
        new_dm_from_json.results[0].data.values, random_numpy_vector1d_float64
    )
