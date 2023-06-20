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
import json

# Third Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber, OneofField
import numpy as np

# First Party
from caikit.core import DataObjectBase, dataobject
from caikit.core.toolkit import error_handler
import alog

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="caikit_data_model.caikit_nlp")
class FloatValue(DataObjectBase):
    float_value: Union[
        Annotated[float, OneofField("data_float"), FieldNumber(1)],
        Annotated[np.float32, OneofField("data_float32"), FieldNumber(2)],
        Annotated[np.float64, OneofField("data_float64"), FieldNumber(3)],
    ]


@dataobject(package="caikit_data_model.caikit_nlp")
class Vector1D(DataObjectBase):
    """Data representation for a 1 dimension vector.
    NOTE: This only support float types
    """

    data: List[FloatValue]

    def _convert_np_to_list(self, values):
        return values.tolist()

    def _convert_to_floatvalue(self, values):

        if isinstance(values, np.ndarray):
            if values.dtype == np.float32:
                return [FloatValue(data_float32=val).to_proto() for val in values]
            if values.dtype == np.float64:
                return [FloatValue(data_float64=val).to_proto() for val in values]
        return [FloatValue(val).to_proto() for val in values]

    def to_dict(self) -> dict:

        # If data is in np.ndarray format, convert it to python list
        if isinstance(self.data, np.ndarray):
            return {"data": self._convert_np_to_list(self.data)}

        return {"data": self.data}

    def fill_proto(self, proto):
        data_proto = getattr(proto, "data")
        data_proto.extend(self._convert_to_floatvalue(self.data))

        return proto

    @classmethod
    def from_proto(cls, proto):
        data = []
        for datum in proto.data:
            value_type = FloatValue.from_proto(datum).which_oneof("float_value")
            data.append(getattr(datum, value_type))
        return cls(**{"data": data})

    @classmethod
    def from_json(cls, json_str):
        """Function to convert json_str to DataBase object

        NOTE: Since JSON doesn't support variety of float formats
              we would only be able to convert all the values to python floats here.
              So there would be a loss of information when converting to and from JSON

        Args:
            json_str: str or dict
                A stringified JSON specification/dict of the data_model

        Returns:
            caikit.core.data_model.DataBase
                A DataBase object.
        """
        error.type_check("<NLP97633962E>", dict, str, json_str=json_str)
        if isinstance(json_str, str):
            json_str = json.loads(json_str)

        error.value_check(
            "<NLP49158301E>", "data" in json_str, "invalid json_str provided!"
        )

        data = []
        for datum in json_str["data"]:
            data.append(datum)
        return cls(data=data)


@dataobject(package="caikit_data_model.caikit_nlp")
class EmbeddingResult(DataObjectBase):
    """Data representation for an embedding matrix holding 2D vectors"""

    data: List[Vector1D]

    def _convert_np_to_vector1d(self, values):
        if values.ndim() == 1:
            return [Vector1D(values)]

        if values.ndim() == 2:
            return [Vector1D(vector) for vector in values]

        error(
            "<NLP18185885E>",
            f"Unsupported dimensions {values.ndim()} for EmbeddingResult",
        )

    def _convert_list_to_vector1d(self, values):
        temp_data = None
        if isinstance(values[0], list):
            temp_data = [Vector1D(vector) for vector in values]
        else:
            temp_data = [Vector1D(values)]

        return temp_data

    def to_dict(self) -> dict:

        vector_2d: Union(List[Vector1D], None) = None

        # if data is in np.ndarray format, convert it into Vector1D
        if isinstance(self.data, np.ndarray):
            vector_2d = self._convert_np_to_vector1d(self.data)
        elif isinstance(self.data, list):
            vector_2d = self._convert_list_to_vector1d(self.data)
        else:
            error(
                "<NLP69094934E>",
                f"Only list and np.ndarray supported currently. Provided {type(self.data)}",
            )

        return {"data": [vector.to_dict() for vector in vector_2d]}

    def fill_proto(self, proto):
        vector_proto = getattr(proto, "data")

        proto_values = None
        if isinstance(self.data, list):
            proto_values = self._convert_list_to_vector1d(self.data)
        elif isinstance(self.data, np.ndarray):
            proto_values = self._convert_np_to_vector1d(self.data)
        else:
            error(
                "<NLP58452707E>",
                "unsupported type for data. Only list and np.ndarray supported",
            )

        vector_proto.extend([val.to_proto() for val in proto_values])

        return proto

    @classmethod
    def from_json(cls, json_str):
        error.type_check("<NLP83742499E>", dict, str, json_str=json_str)

        if isinstance(json_str, str):
            json_str = json.loads(json_str)

        error.value_check(
            "<NLP64229346E>", "data" in json_str, "invalid json_str provided!"
        )

        return cls(data=[Vector1D.from_json(datum) for datum in json_str["data"]])
