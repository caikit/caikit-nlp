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
from caikit.core.exceptions import error_handler
from caikit.interfaces.nlp.data_model import (
    GeneratedTextResult,
    GeneratedTextStreamResult,
    GeneratedToken,
    TokenStreamDetails,
)
from caikit_tgis_backend.protobufs import generation_pb2
import alog

# Local
from ...data_model import ExponentialDecayLengthPenalty
from .model_run_utils import GENERATE_FUNCTION_ARGS, VALID_DECODING_METHODS

log = alog.use_channel("TGIS_UTILS")
error = error_handler.get(log)

GENERATE_FUNCTION_TGIS_ARGS = """
    {}
    preserve_input_text: str
        Whether or not the source string should be contained in the generated output,
        e.g., as a prefix.
""".format(
    GENERATE_FUNCTION_ARGS
)


def validate_inf_params(
    text,
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
    """Validate inference parameters

    Args:
        eos_token: str
           A special token representing the end of a sentence.
        {}
    """.format(
        GENERATE_FUNCTION_TGIS_ARGS
    )
    error.type_check("<NLP65883535E>", str, text=text)
    error.type_check("<NLP65883537E>", bool, preserve_input_text=preserve_input_text)
    error.type_check("<NLP85452188E>", str, allow_none=True, eos_token=eos_token)
    error.type_check(
        "<NLP03860681E>",
        int,
        allow_none=True,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        truncate_input_tokens=truncate_input_tokens,
        top_k=top_k,
        seed=seed,
    )

    error.value_check(
        "<NLP03521352E>",
        max_new_tokens >= min_new_tokens,
        f"Maximum new tokens [{max_new_tokens}] has to be greater than minimum new tokens \
        [{min_new_tokens}]",
    )

    error.value_check(
        "<NLP03521363E>",
        decoding_method in VALID_DECODING_METHODS,
        f"Decoding method [{decoding_method}] not in valid decoding methods: "
        f"[{VALID_DECODING_METHODS}]",
    )
    error.type_check(
        "<NLP55267524E>",
        float,
        allow_none=True,
        top_p=top_p,
        typical_p=typical_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_time=max_time,
    )
    error.type_check(
        "<NLP28185345E>",
        ExponentialDecayLengthPenalty,
        tuple,
        allow_none=True,
        exponential_decay_length_penalty=exponential_decay_length_penalty,
    )

    error.type_check_all(
        "<NLP41311584E>", str, allow_none=True, stop_sequences=stop_sequences
    )

    error.value_check(
        "<NLP28185344E>",
        not temperature or temperature >= 0.05,
        "temperature must be >= 0.05",
    )

    error.value_check(
        "<NLP28185346E>",
        not top_p or 0 < top_p <= 1.0,
        "top_p must be > 0.0 and <= 1.0",
    )

    error.value_check(
        "<NLP28185347E>", not top_k or top_k >= 0, "top_k must be strictly positive"
    )

    error.value_check(
        "<NLP28185348E>", not typical_p or typical_p <= 1.0, "typical_p must be <= 1.0"
    )

    error.value_check(
        "<NLP28185349E>",
        not repetition_penalty or repetition_penalty > 0.0,
        "repetition_penalty must be > 0.0",
    )

    if exponential_decay_length_penalty:
        if isinstance(exponential_decay_length_penalty, ExponentialDecayLengthPenalty):
            exponential_decay_length_penalty = (
                exponential_decay_length_penalty.start_index,
                exponential_decay_length_penalty.decay_factor,
            )
        error.value_check(
            "<NLP28185350E>",
            exponential_decay_length_penalty[1] >= 1.0
            and exponential_decay_length_penalty[1] <= 10.0,
            "decay_factor in exponential_decay_length_penalty must be >= 1.0 and <= 10.0",
        )

    if decoding_method == "GREEDY" and (
        temperature not in (1, None)
        or top_k not in (0, None)
        or top_p not in (1, None)
        or seed
    ):
        raise ValueError(
            "sampling parameters (temperature/top_k/top_p/typical_p/seed) aren't "
            "applicable in greedy decoding mode"
        )


def get_params(
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
):
    """Get generation parameters

    Args:
        {}
    """.format(
        GENERATE_FUNCTION_TGIS_ARGS
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
        stop_sequences=stop_sequences,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        time_limit_millis=int(max_time * 1000) if max_time else None,
    )

    if exponential_decay_length_penalty:
        if isinstance(exponential_decay_length_penalty, ExponentialDecayLengthPenalty):
            exponential_decay_length_penalty = (
                exponential_decay_length_penalty.start_index,
                exponential_decay_length_penalty.decay_factor,
            )
        exponential_decay_length_penalty = (
            generation_pb2.DecodingParameters.LengthPenalty(
                start_index=exponential_decay_length_penalty[0],
                decay_factor=exponential_decay_length_penalty[1],
            )
        )

    decoding_parameters = generation_pb2.DecodingParameters(
        repetition_penalty=repetition_penalty,
        length_penalty=exponential_decay_length_penalty,
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
            {}
        Returns:
            GeneratedTextResult
                Generated text result produced by TGIS.
        """.format(
            GENERATE_FUNCTION_TGIS_ARGS
        )

        # In case internal client is not configured - generation
        # cannot be done (individual modules may already check
        # for this)
        error.value_check(
            "<NLP72700256E>",
            self.tgis_client is not None,
            "Backend must be configured and loaded for generate",
        )

        validate_inf_params(
            text=text,
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

        log.debug("Building protobuf request to send to TGIS")

        params = get_params(
            preserve_input_text=preserve_input_text,
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
            seed=seed,
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
            {}
        Returns:
            Iterable[GeneratedTextStreamResult]
        """.format(
            GENERATE_FUNCTION_TGIS_ARGS
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

        validate_inf_params(
            text=text,
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

        params = get_params(
            preserve_input_text=preserve_input_text,
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
