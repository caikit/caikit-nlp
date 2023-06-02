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
"""This file contains a distributed backend implementation for leveraging the PEFT-trained
prompt vectors in TGIS generation requests.
"""
# Standard
import os

# First Party
from caikit.core import (
    ModuleBase,
    ModuleConfig,
    ModuleSaver,
    block,
    module_backend_config,
)
from caikit.core.module_backends import backend_types
from caikit.core.toolkit import error_handler
from caikit_tgis_backend import TGISBackend
from caikit_tgis_backend.protobufs import generation_pb2
import alog

# Local
from ...data_model.generation import GeneratedResult
from ...toolkits.verbalizer_utils import render_verbalizer
from . import PeftPromptTuning

log = alog.use_channel("PEFT_PROMPT_REMOTE")
error = error_handler.get(log)


@block(backend_type=TGISBackend.backend_type, base_module=PeftPromptTuning)
class PeftPromptTuningTGIS(ModuleBase):
    SUPPORTED_LOAD_BACKENDS = [TGISBackend.backend_type, backend_types.LOCAL]
    ## Module Interface ##

    def __init__(self, base_model_name, prompt_cache_id, eos_token, verbalizer) -> None:
        super().__init__()
        # Configure the internal client
        self._client = module_backend_config.get_backend(
            TGISBackend.backend_type
        ).get_client(base_model_name)
        self.base_model_name = base_model_name
        self._prompt_cache_id = prompt_cache_id
        self.eos_token = eos_token
        self.verbalizer = verbalizer

    @classmethod
    def load(cls, model_path: str) -> "PeftPromptTuningTGIS":
        """Load a TGIS Peft Prompt Tuning distributed module. Note that we do not
        leverage artifacts stored within the model here, and we assume that the
        prompt vector is already available at a place that the TGIS server can pick it
        up.

        Args:
            model_path: str
                Path to the model to be loaded.
        Returns:
            PeftPromptTuningTGIS
                Instance of this class built from the on disk model.
        """
        config = ModuleConfig.load(model_path)
        eos_token = config.eos_token
        verbalizer = config.verbalizer
        dir_name = os.path.split(model_path)[-1]
        # NOTE: base_model_name is used as "model_id" when calling to TGIS backend
        base_model_name = config.get("base_model_name", "")
        prompt_cache_id = dir_name
        error.type_check("<FPT24633932E>", str, prompt_cache_id=prompt_cache_id)
        # NOTE: prompt model config stores a base_model_config
        # which can be used to validate if the prompt is tuned
        # for the model it is being used with. However,
        # we are currently not accessing or assuming the accessibilty of
        # base model, thus not validating.
        # NOTE: When we access base_model_config, we need to make sure
        # we convert make it valid json compatible dict (aka doesn't have non string keys)
        log.debug("Prompt ID: %s", prompt_cache_id)
        log.debug("TGIS model ID: %s", base_model_name)
        return cls(base_model_name, prompt_cache_id, eos_token, verbalizer)

    def save(self, model_path: str):
        """Export the config for this model.

        model_path: str
            Path to which we should write our model.
        """
        saver = ModuleSaver(
            self,
            model_path=model_path,
        )
        with saver:
            saver.update_config(
                {
                    "base_model_name": self.base_model_name,
                    "prompt_cache_id": self._prompt_cache_id,
                    "eos_token": self.eos_token,
                    "verbalizer": self.verbalizer,
                }
            )

    def run(self, text, preserve_input_text=False):
        """Run inference against the model running in TGIS. Currently we leverage greedy decoding
        and apply the same verbalizer used for training the local model prior to sending the
        request to TGIS.

        Args:
            text: str
                Source string to be encoded for generation.
            preserve_input_text: str
                Whether or not the source string should be contained in the generated output,
                e.g., as a prefix.

        Returns:
            GeneratedResult
                Generated text result produced by TGIS.
        """
        verbalized_text = render_verbalizer(self.verbalizer, {"input": text})
        log.debug("Building protobuf request to send to TGIS")
        res_options = generation_pb2.ResponseOptions(
            input_text=preserve_input_text,
            generated_tokens=True,
            input_tokens=False,
            token_logprobs=True,
            token_ranks=True,
        )
        stopping = generation_pb2.StoppingCriteria(
            stop_sequences=[self.eos_token],
        )
        params = generation_pb2.Parameters(
            response=res_options,
            stopping=stopping,
        )

        gen_reqs = [generation_pb2.GenerationRequest(text=verbalized_text)]
        request = generation_pb2.BatchedGenerationRequest(
            requests=gen_reqs,
            model_id=self.base_model_name,
            prefix_id=self._prompt_cache_id,
            params=params,
        )

        # Currently, we send a batch request of len(x)==1, so we expect one response back
        with alog.ContextTimer(log.trace, "TGIS request duration: "):
            batch_response = self._client.Generate(request)

        error.value_check(
            "<FPT12333421E>",
            len(batch_response.responses) == 1,
            f"Got {len(batch_response.responses)} responses for a single request",
        )
        response = batch_response.responses[0]

        return GeneratedResult(
            generated_token_count=response.generated_token_count,
            text=response.text,
            stop_reason=response.stop_reason,
            producer_id=self.PRODUCER_ID,
        )
