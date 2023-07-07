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
import os

# Third Party
from transformers import AutoConfig

# First Party
from caikit.core.data_model import DataStream
from caikit.core.module_backends import BackendBase, backend_types
from caikit.core.modules import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.core.toolkit import error_handler
from caikit_tgis_backend import TGISBackend
from caikit_tgis_backend.protobufs import generation_pb2
import alog

# Local
from ...data_model import GeneratedResult, GenerationTrainRecord
from ...resources.pretrained_model import (
    HFAutoCausalLM,
    HFAutoSeq2SeqLM,
    PretrainedModelBase,
)
from .text_generation_task import TextGenerationTask

log = alog.use_channel("TXT_GEN")
error = error_handler.get(log)


# pylint: disable=too-many-lines,too-many-instance-attributes
@module(
    id="f9181353-4ccf-4572-bd1e-f12bcda26792",
    name="Text Generation",
    version="0.1.0",
    task=TextGenerationTask,
)
class FineTuning(ModuleBase):
    """Module to provide fine-tuning support for text generation task"""

    def __init__(self):
        super().__init__()

    @classmethod
    def train(
        cls,
        base_model: str,  # TODO: Union[str, PretrainedModelBase]
        train_stream: DataStream[GenerationTrainRecord],):


        ## Generate data loader from stream

        ## Fetch trainer from resource

        ## Call Trainer.train function



