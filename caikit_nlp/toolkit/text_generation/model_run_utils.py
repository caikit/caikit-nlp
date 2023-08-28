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
from typing import Optional

# Third Party
from transformers import AutoTokenizer
import torch

# First Party
from caikit.core.data_model.producer import ProducerId
from caikit.interfaces.nlp.data_model import (
    GeneratedTextResult,
    GeneratedTextStreamResult,
)
from caikit.core.toolkit.errors import error_handler
import alog

log = alog.use_channel("RUN_UTILS")
error = error_handler.get(log)

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
    repetition_penalty: float
        The more a token is used within generation the more it is penalized
        to not be picked in successive generation passes.
        Default: 0.0 - means no penalty - equivalent to 1.0
"""

def generate_text_func(
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
    repetition_penalty: Optional[float] = 0.0,
    **kwargs
):
    __doc__ = """
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
    """.format(GENERATE_FUNCTION_ARGS)

    error.type_check("<NLP85452187E>", str, eos_token=eos_token)
    error.type_check("<NLP65883534E>", str, text=text)
    error.type_check("<NLP03860680E>", int, allow_none=True, max_new_tokens=max_new_tokens)
    error.type_check("<NLP30091276E>", int, allow_none=True, min_new_tokens=min_new_tokens)
    error.type_check("<NLP55411551E>", int, allow_none=True, truncate_input_tokens=truncate_input_tokens)
    error.type_check("<NLP84635843E>", int, allow_none=True, top_k=top_k)
    error.type_check("<NLP55267523E>", float, allow_none=True, top_p=top_p)
    error.type_check("<NLP13670202E>", float, allow_none=True, typical_p=typical_p)
    error.type_check("<NLP11929418E>", float, allow_none=True, repetition_penalty=repetition_penalty)

    # NOTE: below is to match TGIS API, where 0 identifies as no truncation
    if truncate_input_tokens == 0:
        # NOTE: below will make model throw error in case inputs are longer
        # than allowed length
        truncation = False

    else:
        truncation = True

    if repetition_penalty == 0.0:
        repetition_penalty = 1.0


    gen_optional_params = {}

    # TODO: Make decoding parameters enums
    if decoding_method == "SAMPLING":
        gen_optional_params["do_sample"] = True
        gen_optional_params["top_k"] = top_k
        gen_optional_params["top_p"] = top_p
        gen_optional_params["typical_p"] = typical_p
        gen_optional_params["temperature"] = temperature

    tok_tensors = tokenizer(
            text,
            truncation=truncation,
            max_length=truncate_input_tokens,
            return_tensors="pt",
        )
    inputs = {k: v.to(model.device) for k, v in tok_tensors.items()}
    with torch.no_grad():
        generate_ids = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            use_cache=True,
            **gen_optional_params,
            **kwargs,
        )

    token_count = generate_ids.size(1) - 1

    preds = [
        tokenizer.decode(
            g, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        for g in generate_ids
    ]

    if generate_ids[0][-1].item() == eos_token:
        finish_reason = "EOS_TOKEN"
    elif generate_ids.size(1) - 1 == max_new_tokens:
        finish_reason = "MAX_TOKENS"
    else:
        finish_reason = "OTHER"
    return GeneratedTextResult(
        generated_tokens=token_count,
        generated_text=preds[0],
        finish_reason=finish_reason,
        producer_id=producer_id,
    )
