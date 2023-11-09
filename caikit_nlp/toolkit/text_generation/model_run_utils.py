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

"""Utility functions used for executing run function for text_generation"""

# Standard
from typing import List, Optional, Tuple, Union

# Third Party
from peft.peft_model import PeftModel
from transformers import AutoModel, AutoTokenizer, StoppingCriteria, TextStreamer
import numpy as np
import torch

# First Party
from caikit.core.data_model.producer import ProducerId
from caikit.core.exceptions import error_handler
from caikit.interfaces.nlp.data_model import (
    FinishReason,
    GeneratedTextResult,
    GeneratedTextStreamResult,
    TokenStreamDetails,
)
import alog

# Local
from caikit_nlp.data_model import ExponentialDecayLengthPenalty

log = alog.use_channel("RUN_UTILS")
error = error_handler.get(log)

VALID_DECODING_METHODS = ["GREEDY", "SAMPLING"]

GENERATE_FUNCTION_ARGS = """
    text: str
        Input string to be used to the generation model.
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
        Default: 1.0 - means disabled - 0.0 equivalent to 1.0
    typical_p: float
        Local typicality measures how similar the conditional probability of
        predicting a target token next is to the expected conditional
        probability of predicting a random token next, given the partial text
        already generated. If set to float < 1, the smallest set of the most
        locally typical tokens with probabilities that add up to typical_p
        or higher are kept for generation. Only applicable when decoding_method
        is SAMPLING.
        Default: 1.0 - means disabled - 0.0 equivalent to 1.0
    temperature: float
        The value used to modulate the next token probabilities.
        Only applicable when decoding_method is SAMPLING.
        Default: 1.0 - means disabled - equivalent to 1.0
    repetition_penalty: float
        The more a token is used within generation the more it is penalized
        to not be picked in successive generation passes.
        Default: 1.0 - means no penalty - 0.0 equivalent to 1.0
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
    stop_sequences: List[str]
        List of strings to be used as stopping criteria
    seed: numpy.uint64
        Random seed to control sampling. Only applicable when decoding_method
        is SAMPLING. Default: None
"""


class Streamer(TextStreamer):
    # The default TextStreamer currently prints to stdout
    # so we override that here
    def on_finalized_text(self, text: str, stream_end: bool = False):
        pass


class SequenceStoppingCriteria(StoppingCriteria):
    # pylint: disable-next=super-init-not-called # false positive: StoppingCriteria is an abc and has no __init__
    def __init__(self, target_sequence_ids):
        self.target_sequence_ids = target_sequence_ids

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the target sequence appears in the generated text
        for seq_id in self.target_sequence_ids:
            if seq_id in input_ids:
                return True  # Stop generation

        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self


def generate_text_func(
    model: "Union[PeftModel, AutoModel]",
    tokenizer: "AutoTokenizer",
    producer_id: ProducerId,
    eos_token: Optional[str],
    text: str,
    max_new_tokens: Optional[int] = 20,
    min_new_tokens: Optional[int] = 0,
    truncate_input_tokens: Optional[int] = 0,
    decoding_method: Optional[str] = "GREEDY",
    top_k: Optional[int] = 0,
    top_p: Optional[float] = 1.0,
    typical_p: Optional[float] = 1.0,
    temperature: Optional[float] = 1.0,
    seed: Optional[np.uint64] = None,
    repetition_penalty: Optional[float] = 1.0,
    max_time: Optional[float] = None,
    exponential_decay_length_penalty: Optional[
        Union[Tuple[int, float], ExponentialDecayLengthPenalty]
    ] = None,
    stop_sequences: Optional[List[str]] = None,
    preserve_input_text: Optional[bool] = True,
    task_type: Optional[str] = None,
    **kwargs,
):
    """
        Args:
            model: PeftModel or transformers.AutoModel
                Peft model or Transformers model
            tokenizer: AutoTokenizer
                Tokenizer to be used with the model
            producer_id: ProducerId
                Caikit producer id associated with the module
            eos_token: str
                End of sequence token to be used with generation
            preserve_input_text: bool
                Applicable only for CAUSAL_LM task type.
                Whether or not the source string should be contained in the generated output,
                e.g., as a prefix. Default True. (Source string will appear as prefix)
            task_type: str or None
                Task type such as CAUSAL_LM, SEQ_2_SEQ_LM, SEQ_CLS or None
            {}
        Returns:
            GeneratedTextResult
    """.format(
        GENERATE_FUNCTION_ARGS
    )

    error.type_check("<NLP85452187E>", str, allow_none=True, eos_token=eos_token)
    error.type_check("<NLP65883534E>", str, text=text)

    error.type_check(
        "<NLP55411551E>",
        int,
        allow_none=True,
        truncate_input_tokens=truncate_input_tokens,
    )

    # NOTE: below is to match TGIS API, where 0 identifies as no truncation
    truncation = truncate_input_tokens != 0

    tok_tensors = tokenizer(
        text,
        truncation=truncation,
        max_length=truncate_input_tokens,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in tok_tensors.items()}

    input_token_count = tok_tensors["input_ids"].size(1)

    gen_optional_params = __process_gen_args(
        tokenizer,
        max_new_tokens,
        min_new_tokens,
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
    )

    if "attention_mask" in inputs:
        gen_optional_params["attention_mask"] = inputs["attention_mask"]

    # NOTE: Below is required as `task_id` is a required field for generation
    # with MPT in PEFT. We are manually setting task id to 0 vector since
    # we do not allow setting task specific id anyways.
    if isinstance(model, PeftModel):
        gen_optional_params["task_ids"] = torch.zeros(
            inputs["input_ids"].shape[0], dtype=inputs["input_ids"].dtype
        ).to(model.device)

    with torch.no_grad():
        generate_ids = model.generate(
            input_ids=inputs["input_ids"],
            **gen_optional_params,
            **kwargs,
        )

    token_count = generate_ids.size(1) - 1

    preds = [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for g in generate_ids
    ]

    if preserve_input_text is not True:
        generated_text = __postprocess_remove_input_text(
            tokenizer, preds, inputs, task_type
        )
    else:
        generated_text = preds[0]

    if (eos_token and tokenizer.decode(generate_ids[0, -1].item()) == eos_token) or (
        generate_ids[0, -1] == tokenizer.eos_token_id
    ):
        finish_reason = FinishReason.EOS_TOKEN
    elif ("stopping_criteria" in gen_optional_params) and (
        gen_optional_params["stopping_criteria"](
            generate_ids,
            None,  # scores, unused by SequenceStoppingCriteria
        )
    ):
        finish_reason = FinishReason.STOP_SEQUENCE
    else:
        finish_reason = FinishReason.MAX_TOKENS

    return GeneratedTextResult(
        generated_tokens=token_count,
        generated_text=generated_text,
        finish_reason=finish_reason,
        producer_id=producer_id,
        input_token_count=input_token_count,
        seed=seed,
    )


def __postprocess_remove_input_text(tokenizer, preds, inputs, task_type):
    """For Causal LM task types, preserve_input_text set to False will
    remove the input text from generated output.
    """
    if task_type == "CAUSAL_LM":
        prompt_length = len(
            tokenizer.decode(
                inputs["input_ids"][0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        )
        generated_text = preds[0][prompt_length:]
    else:
        log.warning(
            "<NLP16125792W>",
            f"preserve_input_text flag is not applicable for task type {task_type}. \
              Returning model generated prediction",
        )
        generated_text = preds[0]
    return generated_text


def generate_text_func_stream(
    model,
    tokenizer,
    producer_id: ProducerId,
    eos_token: str,
    text: str,
    max_new_tokens: Optional[int] = 20,
    min_new_tokens: Optional[int] = 0,
    truncate_input_tokens: Optional[int] = 0,
    decoding_method: Optional[str] = "GREEDY",
    top_k: Optional[int] = 0,
    top_p: Optional[float] = 0.0,
    typical_p: Optional[float] = 0.0,
    temperature: Optional[float] = 1.0,
    seed: Optional[np.uint64] = None,
    repetition_penalty: Optional[float] = 0.0,
    max_time: Optional[float] = None,
    exponential_decay_length_penalty: Optional[
        Union[Tuple[int, float], ExponentialDecayLengthPenalty]
    ] = None,
    stop_sequences: Optional[List[str]] = None,
    **kwargs,
):
    """
        Args:
            model: PeftModel or transformers.AutoModel
                Peft model or Transformers model
            tokenizer: AutoTokenizer
                Tokenizer to be used with the model
            producer_id: ProducerId
                Caikit producer id associated with the module
            eos_token: str
                End of sequence token to be used with generation
            {}
        Returns:
            GeneratedTextResult
    """.format(
        GENERATE_FUNCTION_ARGS
    )
    error.type_check("<NLP53933302E>", str, eos_token=eos_token)
    error.type_check("<NLP61673437E>", str, text=text)

    error.type_check(
        "<NLP60192564E>",
        int,
        allow_none=True,
        truncate_input_tokens=truncate_input_tokens,
    )

    # NOTE: below is to match TGIS API, where 0 identifies as no truncation
    truncation = truncate_input_tokens != 0

    tok_tensors = tokenizer(
        text,
        truncation=truncation,
        max_length=truncate_input_tokens,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in tok_tensors.items()}

    input_token_count = len(tok_tensors)

    streamer = Streamer(tokenizer)

    gen_optional_params = __process_gen_args(
        tokenizer,
        max_new_tokens,
        min_new_tokens,
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
    )

    with torch.no_grad():
        # Run tokenized tensors through the rest of the PEFT model
        stream_outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            streamer=streamer,
            **gen_optional_params,
            **kwargs,
        )
        details = TokenStreamDetails(
            input_token_count=input_token_count,
            seed=seed,
        )
        for stream_part in stream_outputs:
            gen_text = tokenizer.batch_decode(
                stream_part.detach().cpu().numpy(), skip_special_tokens=True
            )
            yield GeneratedTextStreamResult(
                generated_text=gen_text, details=details, producer_id=producer_id
            )


def __process_gen_args(
    tokenizer,
    max_new_tokens,
    min_new_tokens,
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
    """Utility function to preprocess model generate arguments"""
    error.type_check(
        "<NLP03860680E>", int, allow_none=True, max_new_tokens=max_new_tokens
    )
    error.type_check(
        "<NLP30091276E>", int, allow_none=True, min_new_tokens=min_new_tokens
    )
    error.type_check("<NLP84635843E>", int, allow_none=True, top_k=top_k)
    error.type_check("<NLP55267523E>", float, allow_none=True, top_p=top_p)
    error.type_check(
        "<NLP13670202E>",
        float,
        allow_none=True,
        typical_p=typical_p,
        temperature=temperature,
    )
    error.type_check(
        "<NLP11929418E>",
        float,
        allow_none=True,
        repetition_penalty=repetition_penalty,
        max_time=max_time,
    )
    error.type_check_all(
        "<NLP41311583E>", str, allow_none=True, stop_sequences=stop_sequences
    )
    error.type_check("<NLP28185342E>", int, allow_none=True, seed=seed)

    error.value_check(
        "<NLP80772084E>",
        max_new_tokens >= min_new_tokens,
        "Max new tokens needs to be bigger than min new tokens",
    )

    if isinstance(exponential_decay_length_penalty, ExponentialDecayLengthPenalty):
        exponential_decay_length_penalty = (
            exponential_decay_length_penalty.start_index,
            exponential_decay_length_penalty.decay_factor,
        )

    error.type_check(
        "<NLP81276841E>",
        tuple,
        allow_none=True,
        exponential_decay_length_penalty=exponential_decay_length_penalty,
    )

    error.value_check(
        "<NLP03521360E>",
        decoding_method in VALID_DECODING_METHODS,
        f"Decoding method [{decoding_method}] not in valid decoding methods: "
        f"[{VALID_DECODING_METHODS}]",
    )

    if repetition_penalty == 0.0:
        repetition_penalty = 1.0

    gen_optional_params = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "repetition_penalty": repetition_penalty,
        "use_cache": True,
        "max_time": max_time,
        "exponential_decay_length_penalty": exponential_decay_length_penalty,
    }

    # TODO: Make decoding parameters enums
    if decoding_method == "SAMPLING":
        gen_optional_params["do_sample"] = True
        gen_optional_params["top_k"] = top_k
        gen_optional_params["top_p"] = top_p
        gen_optional_params["typical_p"] = typical_p
        gen_optional_params["temperature"] = temperature
        gen_optional_params["seed"] = seed

    if stop_sequences and len(stop_sequences) > 0:
        # Tokenize sequences
        stop_sequence_ids = tokenizer.encode(stop_sequences)
        stopping_criteria = SequenceStoppingCriteria(stop_sequence_ids)
        gen_optional_params["stopping_criteria"] = stopping_criteria

    return gen_optional_params
