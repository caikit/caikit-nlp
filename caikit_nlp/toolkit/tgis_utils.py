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
"""This file is for helper functions related to TGIS.
"""
# First Party
from caikit_tgis_backend.protobufs import generation_pb2


def get_params(preserve_input_text, eos_token, max_new_tokens, min_new_tokens):
    """

     Args:
        preserve_input_text: str
            Whether or not the source string should be contained in the generated output,
            e.g., as a prefix.
        eos_token: str
            A special token representing the end of a sentence.
        max_new_tokens: int
            The maximum numbers of tokens to generate.
        min_new_tokens: int
            The minimum numbers of tokens to generate.
    """
    res_options = generation_pb2.ResponseOptions(
        input_text=preserve_input_text,
        generated_tokens=True,
        input_tokens=False,
        token_logprobs=True,
        token_ranks=True,
    )
    stopping = generation_pb2.StoppingCriteria(
        stop_sequences=[eos_token],
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
    )
    params = generation_pb2.Parameters(
        response=res_options,
        stopping=stopping,
    )
    return params