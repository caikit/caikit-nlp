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

# Local
from ..data_model import ExponentialDecayLengthPenalty

log = alog.use_channel("TGIS_UTILS")
error = error_handler.get(log)

VALID_DECODING_METHODS = ["GREEDY", "SAMPLING"]

# pylint: disable=duplicate-code

GENERATE_FUNCTION_ARGS = """
    text: str
        Input string to be used to the generation model.
    preserve_input_text: str
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
        Default: 0 - means don't truncate, thus throw error.
    decoding_method: str
        Parameters for conditionally penalizing / boosting
        candidate tokens during decoding.
        Options: "GREEDY" (default), "SAMPLING"
    top_k: int
        The number of highest probability vocabulary tokens to keep for
        top-k-filtering. Only applicable when decoding_method is SAMPLING.
        Default: 0 - means disabled
    top_p: float
        If set to float < 1, only the smallest set of most probable tokens
        with probabilities that add up to top_p or higher are kept for
        generation. Only applicable when decoding_method is SAMPLING.
        Default: 0.0 - means disabled - equivalent to 1.0
    typical_p: float
        Local typicality measures how similar the conditional probability of
        predicting a target token next is to the expected conditional
        probability of predicting a random token next, given the partial text
        already generated. If set to float < 1, the smallest set of the most
        locally typical tokens with probabilities that add up to typical_p
        or higher are kept for generation. Only applicable when decoding_method
        is SAMPLING.
        Default: 0.0 - means disabled - equivalent to 1.0
    temperature: float
        The value used to modulate the next token probabilities.
        Only applicable when decoding_method is SAMPLING.
        Default: 1.0 - means disabled - equivalent to 1.0
    seed: int
        Random seed to control sampling. Only applicable when decoding_method
        is SAMPLING. Default: None
    repetition_penalty: float
        The more a token is used within generation the more it is penalized
        to not be picked in successive generation passes.
        Default: 0.0 - means no penalty - equivalent to 1.0
    max_time: float
        Amount of time in seconds that the query should take maximum.
        NOTE: this does not include network overhead.
        Range: 0-120.0
    exponential_decay_length_penalty: Tuple(int, float)
        This Tuple adds an exponentially increasing length penalty, after
        a certain amount of tokens have been generated. The tuple shall
        consist of: (start_index, decay_factor) where start_index
        indicates where penalty starts and decay_factor represents the factor
        of exponential decay
    stop_sequences: List[str]:
        List of strings to be used as stopping criteria
"""


def get_params(
    preserve_input_text,
    eos_token,
    max_new_tokens,
    min_new_tokens,
    truncate_input_tokens,
    decoding_method,
    top_k,
    top_p,
    typical_p,
    temperature,
    seed,
    repetition_penalty,
    max_time,
    exponential_decay_length_penalty,
    stop_sequences,
):
    """Get generation parameters

    Args:
        preserve_input_text: str
           Whether or not the source string should be contained in the generated output,
           e.g., as a prefix.
        eos_token: str
           A special token representing the end of a sentence.
        {}
    """.format(
        GENERATE_FUNCTION_ARGS
    )

    if decoding_method == "GREEDY":
        decoding = generation_pb2.DecodingMethod.GREEDY
    elif decoding_method == "SAMPLING":
        decoding = generation_pb2.DecodingMethod.SAMPLE

    sampling_parameters = generation_pb2.SamplingParameters(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        typical_p=typical_p,
        seed=seed,
    )

    res_options = generation_pb2.ResponseOptions(
        input_text=preserve_input_text,
        generated_tokens=True,
        input_tokens=False,
        token_logprobs=True,
        token_ranks=True,
    )
    stopping = generation_pb2.StoppingCriteria(
        stop_sequences=stop_sequences or [eos_token] if eos_token else None,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        time_limit_millis=max_time,
    )

    start_index = None
    decay_factor = None

    if exponential_decay_length_penalty:
        if isinstance(exponential_decay_length_penalty, tuple):
            start_index = exponential_decay_length_penalty[0]
            decay_factor = exponential_decay_length_penalty[1]
        elif isinstance(
            exponential_decay_length_penalty, ExponentialDecayLengthPenalty
        ):
            start_index = exponential_decay_length_penalty.start_index
            decay_factor = exponential_decay_length_penalty.decay_factor

    length_penalty = generation_pb2.DecodingParameters.LengthPenalty(
        start_index=start_index, decay_factor=decay_factor
    )

    decoding_parameters = generation_pb2.DecodingParameters(
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
    )

    params = generation_pb2.Parameters(
        method=decoding,
        sampling=sampling_parameters,
        response=res_options,
        stopping=stopping,
        decoding=decoding_parameters,
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
        decoding_method,
        top_k,
        top_p,
        typical_p,
        temperature,
        seed,
        repetition_penalty,
        max_time,
        exponential_decay_length_penalty,
        stop_sequences,
    ) -> GeneratedTextResult:
        """Generate unary output from model in TGIS

        Args:
            text: str
                Source string to be encoded for generation.
            preserve_input_text: bool
                Whether or not the source string should be contained in the generated output,
                e.g., as a prefix.
            {}
        Returns:
            GeneratedTextResult
                Generated text result produced by TGIS.
        """.format(
            GENERATE_FUNCTION_ARGS
        )

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
            decoding_method=decoding_method,
            top_k=top_k,
            top_p=top_p,
            typical_p=typical_p,
            temperature=temperature,
            seed=seed,
            repetition_penalty=repetition_penalty,
            max_time=max_time,
            exponential_decay_length_penalty=exponential_decay_length_penalty,
            stop_sequences=stop_sequences,
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
            input_token_count=response.input_token_count,
        )

    def stream_generate(
        self,
        text,
        preserve_input_text,
        max_new_tokens,
        min_new_tokens,
        truncate_input_tokens,
        decoding_method,
        top_k,
        top_p,
        typical_p,
        temperature,
        seed,
        repetition_penalty,
        max_time,
        exponential_decay_length_penalty,
        stop_sequences,
    ) -> Iterable[GeneratedTextStreamResult]:
        """Generate stream output from model in TGIS

        Args:
            text: str
                Source string to be encoded for generation.
            preserve_input_text: bool
                Whether or not the source string should be contained in the generated output,
                e.g., as a prefix.
            {}
        Returns:
            Iterable[GeneratedTextStreamResult]
        """.format(
            GENERATE_FUNCTION_ARGS
        )

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
            decoding_method=decoding_method,
            top_k=top_k,
            top_p=top_p,
            typical_p=typical_p,
            temperature=temperature,
            seed=seed,
            repetition_penalty=repetition_penalty,
            max_time=max_time,
            exponential_decay_length_penalty=exponential_decay_length_penalty,
            stop_sequences=stop_sequences,
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
                input_token_count=stream_part.input_token_count,
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
