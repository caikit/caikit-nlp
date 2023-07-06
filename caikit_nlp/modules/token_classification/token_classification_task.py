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
"""This task can be promoted to caikit/caikit for wider usage when applicable
to multiple modules
"""
# Standard
from typing import Iterable

# First Party
from caikit.core import TaskBase, task

# Local
from ...data_model import StreamingTokenClassificationResult, TokenClassificationResult


@task(
    unary_parameters={"text": str},
    streaming_parameters={"text": Iterable[str]},
    unary_output_type=TokenClassificationResult,
    streaming_output_type=Iterable[StreamingTokenClassificationResult],
)
class TokenClassificationTask(TaskBase):
    pass
