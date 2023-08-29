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
"""Data structures for text generation representations
"""
# Standard
from enum import Enum
from typing import List

# First Party
from caikit.core import DataObjectBase

# First party
import alog
import caikit

log = alog.use_channel("DATAM")


class PromptOutputModelType(Enum):
    ENCODER = "ENCODER"
    DECODER = "DECODER"


@caikit.core.dataobject(package="caikit_data_model.caikit_nlp")
class GenerationTrainRecord(DataObjectBase):
    input: str
    output: str


@caikit.core.dataobject(package="caikit_data_model.caikit_nlp")
class TuningConfig(DataObjectBase):
    # If prompt_tuning_init_text is not provided, then random would be used
    # but since random is not supported currently, we want to keep text to be
    # required
    num_virtual_tokens: int
    # TODO: Move all _init_ params to separate object
    prompt_tuning_init_text: str
    prompt_tuning_init_method: str
    #    could be: `RANDOM`, `TEXT`, `ONLY_SOURCE_SHARED` and `AVERAGE_SOURCE`
    #
    prompt_tuning_init_source_model: str  # this maps to prompt_tuning_init_state_dict_path in MPT
    #     which is path pointing to the state dict of the model to be used for initialization
    #
    # token_dim: int # Optional - dimension of the virtual tokens.
    #
    output_model_types: List[str]
    # this replaces `num_transformer_submodules`
    #     option and can take values encoder, decoder. For each
    #     selected resource type will only provide certain possibilities,
    #     for example, causal-lm models will only provide decoder as option.
    #     If None provided, then we will use defaults for that model_type
    # num_transformer_submodules: int # Optional - The number of transformer submodules in the
    # base transformer model.
    #     1 for all decoder-only models and 2 for encoder-decoder models.
    #     If 1 is used for encoder-decoder models, the prompt will be used for the encoder only.
    #
    # num_attention_heads: int # Optional - The number of attention heads in the
    # base transformer model
    #
    # num_layers: int # Optional - The number of layers in the base transformer model
    #
    # encoder_hidden_size: int # Optional -  The hidden size of the prompt encoder.


@caikit.core.dataobject(package="caikit_data_model.caikit_nlp")
class ExponentialDecayLengthPenalty(DataObjectBase):
    start_index: int
    decay_factor: float
