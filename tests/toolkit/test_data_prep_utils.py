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

# Standard
from unittest import mock

# First Party
import caikit

# Third Party
import transformers
import torch

# Local
from caikit_nlp.data_model import GenerationTrainRecord
from caikit_nlp.resources.pretrained_model import HFAutoCausalLM, HFAutoSeq2SeqLM
from caikit_nlp.toolkit.data_prep_utils import build_tokenize_function
from tests.fixtures import (
    causal_lm_dummy_model,
    causal_lm_train_kwargs,
)

SAMPLE_TRAINING_DATA = caikit.core.data_model.DataStream.from_iterable([
    GenerationTrainRecord(input="This is a sequence", output="This is a target"),
    GenerationTrainRecord(input="I am just a little", output="bit tired"),
])

### Tests for Causal Language Modeling tokenization
def test_causal_lm_tokenize_func_contains_wrapped_stream(causal_lm_dummy_model):
    """Ensure the Causal LM tokenize func produces a wrapped stream that can be flattened."""
    (tok_func, requires_unwrapping) = build_tokenize_function(
        tokenizer=causal_lm_dummy_model.tokenizer,
        max_source_length=100,
        max_target_length=100,
        verbalizer="{{input}}",
        task_type=HFAutoCausalLM.TASK_TYPE
    )
    tok_res = tok_func(GenerationTrainRecord(input="hello", output="world"))
    map_stream = SAMPLE_TRAINING_DATA.map(tok_func)
    # Since tok_func for causal lm creates a datastream, we should get a stream
    # back; make sure that it's a stream of streams before proceeding
    assert requires_unwrapping is True
    assert isinstance(tok_res, caikit.core.data_model.DataStream)
    assert isinstance(map_stream.peek(), caikit.core.data_model.DataStream)
    # Ensure that we can unwrap the stream
    unwrapped_stream = map_stream.flatten()
    assert isinstance(unwrapped_stream.peek(), transformers.tokenization_utils_base.BatchEncoding)

def test_causal_lm_output_correctness(causal_lm_dummy_model):
    """Validate the correctness of the attention mask for the language modeling objective."""
    sample = GenerationTrainRecord(input="This len does not matter", output="but this one does!")
    (tok_func, requires_unwrapping) = build_tokenize_function(
        tokenizer=causal_lm_dummy_model.tokenizer,
        max_source_length=100,
        max_target_length=100,
        verbalizer="{{input}}",
        task_type=HFAutoCausalLM.TASK_TYPE
    )
    input_tok = causal_lm_dummy_model.tokenizer.encode(sample.input)
    output_tok = causal_lm_dummy_model.tokenizer.encode(sample.output)
    tok_stream = tok_func(sample)
    # Ensure we get one token per output in our stream
    assert isinstance(tok_stream, caikit.core.data_model.DataStream)
    assert len(tok_stream) == len(output_tok)
    for idx, tok_sample in enumerate(tok_stream):
        # We expect by default, everything is in order, and each attention mask grows the tokens
        # we attend to in the target by one, until we are paying attention to the whole sequence.
        expected_target_mask = torch.tensor(([1] * (idx + 1)) + [0] * (len(output_tok) - idx - 1))
        actual_target_mask = torch.tensor(tok_sample["attention_mask"][-len(output_tok):])
        assert bool(torch.all(expected_target_mask == actual_target_mask))
        # Check the source mask; we should always attend to the whole source sequence
        actual_source_mask = torch.tensor(tok_sample["attention_mask"][:len(input_tok)])
        assert bool(torch.all(torch.tensor([1]*len(input_tok)) == actual_source_mask))
        # Also, the number of tokens we attend to should be the sum of toks in input/output
        assert (len(actual_target_mask) + len(actual_source_mask)) == len(tok_sample["attention_mask"])
