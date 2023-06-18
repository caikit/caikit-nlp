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
from caikit.core.toolkit import error_handler
import alog

log = alog.use_channel("DATAM")
error = error_handler.get(log)

# TODO:
# At this point, py-to-proto==0.2.0 does not support other float types like float16 etc.
# Once those types are supported in py-to-proto, we would want to enable those
# as supported data types for Vector1D as well


@dataobject(package="caikit_data_model.caikit_nlp")
class Vector1D(DataObjectBase):
    # NOTE: This currently only support float types
    data: List[float]

    def _convert_np_to_list(self, values):
        return values.tolist()

    def to_dict(self) -> dict:

        # If data is in np.ndarray format, convert it to python list
        if isinstance(self.data, np.ndarray):
            return {"data": self._convert_np_to_list(self.data)}

        return {"data": self.data}


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
