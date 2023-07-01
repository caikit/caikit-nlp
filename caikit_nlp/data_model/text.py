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
"""These interfaces can be promoted to caikit/caikit for wider usage
when applicable to multiple modules
"""

# Standard
from typing import List

# First Party
from caikit.core import DataObjectBase
import caikit


@caikit.core.dataobject(package="caikit_data_model.caikit_nlp")
class Span(DataObjectBase):
    start: int
    end: int
    text: str


@caikit.core.dataobject(package="caikit_data_model.caikit_nlp")
class Token(DataObjectBase):
    """Tokens here are the basic units of text. Tokens can be characters, words,
    sub-words, or other segments of text or code, depending on the method of
    tokenization chosen or the task being implemented.
    """

    # NOTE: This data model is purposefully created separate from "Span"
    # to provide flexibility for adding more token level constructs in future
    # if needed, like lemma, part of speech etc.
    span: Span


@caikit.core.dataobject(package="caikit_data_model.caikit_nlp")
class TokenizationResult(DataObjectBase):
    tokens: List[Token]
