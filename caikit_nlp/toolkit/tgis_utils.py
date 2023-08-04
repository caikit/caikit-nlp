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
# Standard
from typing import Iterable

# First Party
from caikit.core.toolkit import error_handler
from caikit.interfaces.nlp.data_model import (
    GeneratedTextResult,
    GeneratedTextStreamResult,
    GeneratedToken,
    TokenStreamDetails,
)
from caikit_tgis_backend.protobufs import generation_pb2
import alog

log = alog.use_channel("TGIS_UTILS")
error = error_handler.get(log)


def get_params(
    preserve_input_text,
    eos_token,
    max_new_tokens,
    min_new_tokens,
    truncate_input_tokens,
):
    """Get generation parameters

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
        truncate_input_tokens: int
            Truncate inputs to provided number of tokens.
    """
    res_options = generation_pb2.ResponseOptions(
        input_text=preserve_input_text,
        generated_tokens=True,
        input_tokens=False,
        token_logprobs=True,
        token_ranks=True,
    )
    stopping = generation_pb2.StoppingCriteria(
        stop_sequences=[eos_token] if eos_token is not None else None,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
    )
    params = generation_pb2.Parameters(
        response=res_options,
        stopping=stopping,
        truncate_input_tokens=truncate_input_tokens,
    )
    return params


class TGISGenerationClient:
    """Client for TGIS generation calls"""

    def __init__(
        self, base_model_name, eos_token, tgis_client, producer_id, prefix_id=None
    ):
        self.base_model_name = base_model_name
        self.eos_token = eos_token
        self.tgis_client = tgis_client
        self.producer_id = producer_id
        self.prefix_id = prefix_id

    def unary_generate(
        self,
        text,
        preserve_input_text,
        max_new_tokens,
        min_new_tokens,
        truncate_input_tokens,
    ) -> GeneratedTextResult:
        """Generate unary output from model in TGIS

        Args:
            text: str
                Source string to be encoded for generation.
            preserve_input_text: bool
                Whether or not the source string should be contained in the generated output,
                e.g., as a prefix.
            max_new_tokens: int
                The maximum numbers of tokens to generate.
                Default: 20
            min_new_tokens: int
                The minimum numbers of tokens to generate.
                Default: 0 - means no minimum
            truncate_input_tokens: int
                Truncate inputs to provided number of tokens. This can be
                use to avoid failing due to input being longer than
                configured limits.
                0 - means don't truncate, thus throw error.
        Returns:
            GeneratedTextResult
                Generated text result produced by TGIS.
        """
        # In case internal client is not configured - generation
        # cannot be done (individual modules may already check
        # for this)
        error.value_check(
            "<NLP72700256E>",
            self.tgis_client is not None,
            "Backend must be configured and loaded for generate",
        )

        log.debug("Building protobuf request to send to TGIS")

        params = get_params(
            preserve_input_text=preserve_input_text,
            eos_token=self.eos_token,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            truncate_input_tokens=truncate_input_tokens,
        )

        gen_reqs = [generation_pb2.GenerationRequest(text=text)]
        if not self.prefix_id:
            request = generation_pb2.BatchedGenerationRequest(
                requests=gen_reqs,
                model_id=self.base_model_name,
                params=params,
            )
        else:
            request = generation_pb2.BatchedGenerationRequest(
                requests=gen_reqs,
                model_id=self.base_model_name,
                prefix_id=self.prefix_id,
                params=params,
            )

        # Currently, we send a batch request of len(x)==1, so we expect one response back
        with alog.ContextTimer(log.trace, "TGIS request duration: "):
            batch_response = self.tgis_client.Generate(request)

        error.value_check(
            "<NLP38899018E>",
            len(batch_response.responses) == 1,
            f"Got {len(batch_response.responses)} responses for a single request",
        )
        response = batch_response.responses[0]

        return GeneratedTextResult(
            generated_text=response.text,
            generated_tokens=response.generated_token_count,
            finish_reason=response.stop_reason,
            producer_id=self.producer_id,
        )

    def stream_generate(
        self,
        text,
        preserve_input_text,
        max_new_tokens,
        min_new_tokens,
        truncate_input_tokens,
    ) -> Iterable[GeneratedTextStreamResult]:
        """Generate stream output from model in TGIS

        Args:
            text: str
                Source string to be encoded for generation.
            preserve_input_text: bool
                Whether or not the source string should be contained in the generated output,
                e.g., as a prefix.
            max_new_tokens: int
                Maximum tokens for the model to generate
            min_new_tokens: int
                Minimum tokens for the model to generate
            truncate_input_tokens: int
                Truncate inputs to provided number of tokens. This can be
                use to avoid failing due to input being longer than
                configured limits.
                0 - means don't truncate, thus throw error.

        Returns:
            Iterable[GeneratedTextStreamResult]
        """
        # In case internal client is not configured - generation
        # cannot be done (individual modules may already check
        # for this)
        error.value_check(
            "<NLP77278635E>",
            self.tgis_client is not None,
            "Backend must be configured and loaded for generate",
        )
        log.debug("Building protobuf request to send to TGIS")

        params = get_params(
            preserve_input_text=preserve_input_text,
            eos_token=self.eos_token,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            truncate_input_tokens=truncate_input_tokens,
        )

        gen_req = generation_pb2.GenerationRequest(text=text)

        if not self.prefix_id:
            request = generation_pb2.SingleGenerationRequest(
                request=gen_req,
                model_id=self.base_model_name,
                params=params,
            )
        else:
            request = generation_pb2.SingleGenerationRequest(
                request=gen_req,
                model_id=self.base_model_name,
                prefix_id=self.prefix_id,
                params=params,
            )

        # stream GenerationResponse
        stream_response = self.tgis_client.GenerateStream(request)

        for stream_part in stream_response:
            details = TokenStreamDetails(
                finish_reason=stream_part.stop_reason,
                generated_tokens=stream_part.generated_token_count,
                seed=stream_part.seed,
            )
            token_list = []
            for token in stream_part.tokens:
                token_list.append(
                    GeneratedToken(text=token.text, logprob=token.logprob)
                )
            yield GeneratedTextStreamResult(
                generated_text=stream_part.text,
                tokens=token_list,
                details=details,
            )
