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
"""Data structures for embedding vector representations
"""

# Standard
from typing import List, Union

# Third Party
import numpy as np

# First Party
from caikit.core import DataObjectBase, dataobject
from caikit.core.exceptions import error_handler
import alog

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="caikit_data_model.caikit_nlp")
class PyFloatSequence(DataObjectBase):
    values: List[float]


@dataobject(package="caikit_data_model.caikit_nlp")
class NpFloat32Sequence(DataObjectBase):
    values: List[np.float32]


@dataobject(package="caikit_data_model.caikit_nlp")
class NpFloat64Sequence(DataObjectBase):
    values: List[np.float64]


@dataobject(package="caikit_data_model.caikit_nlp")
class Vector1D(DataObjectBase):
    """Data representation for a 1 dimension vector of float-type data."""

    data: Union[
        PyFloatSequence,
        NpFloat32Sequence,
        NpFloat64Sequence,
    ]

    # Classes to wrap values for one-of
    WRAP_MAP = {
        "data_pyfloatsequence": PyFloatSequence,
        "data_npfloat32sequence": NpFloat32Sequence,
        "data_npfloat64sequence": NpFloat64Sequence,
    }
    WRAP_DEFAULT = PyFloatSequence

    def to_dict(self) -> dict:
        woo = self.which_oneof("data")  # determine which one of to set
        if not woo:
            woo = "data"
            values = self.data
        else:
            values = self.data.values

        return {
            woo: {
                # coerce numpy.ndarray and numpy.float32 into JSON serializable list of floats
                "values": [float(x) for x in values]
            }
        }

    @classmethod
    def from_proto(cls, proto):
        woo_key = "data"
        woo = proto.WhichOneof(woo_key)
        woo_data = getattr(proto, woo)
        return cls(
            **{woo_key: (cls.WRAP_MAP.get(woo, cls.WRAP_DEFAULT)(woo_data.values))}
        )

    def fill_proto(self, proto):

        if hasattr(self.data, "values"):
            values = self.data.values
        else:
            values = self.data

        if len(values) > 0:
            sample = values[0]
            if isinstance(sample, np.float64):
                proto.data_npfloat64sequence.values.extend(
                    list(np.array(values, dtype=np.float64))
                )
            elif isinstance(sample, np.float32):
                proto.data_npfloat32sequence.values.extend(
                    list(np.array(values, dtype=np.float32))
                )
            elif isinstance(sample, float):
                proto.data_pyfloatsequence.values.extend(values)
            else:
                error(
                    "<NLP58452707E>",
                    "unsupported type of data value. Expecting a float type",
                )

        return proto


@dataobject(package="caikit_data_model.caikit_nlp")
class EmbeddingResult(DataObjectBase):
    """Data representation for an embedding matrix holding 2D vectors"""

    data: List[Vector1D]
