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
from typing import Iterable
import os

# First Party
from caikit.core import ModuleBase, ModuleConfig, ModuleSaver, modules
from caikit.core.module_backends import BackendBase, backend_types
from caikit.core.toolkit import error_handler
from caikit.interfaces.nlp.data_model import (
    GeneratedTextResult,
    GeneratedTextStreamResult,
    GeneratedToken,
    TokenStreamDetails,
)
from caikit.interfaces.nlp.tasks import TextGenerationTask
from caikit_tgis_backend import TGISBackend
from caikit_tgis_backend.protobufs import generation_pb2
import alog

# Local
from ...toolkit.tgis_utils import get_params
from ...toolkit.verbalizer_utils import render_verbalizer
from . import PeftPromptTuning

log = alog.use_channel("PEFT_PROMPT_REMOTE")
error = error_handler.get(log)


@modules.module(backend_type=TGISBackend.backend_type, base_module=PeftPromptTuning)
class PeftPromptTuningTGIS(ModuleBase):
    SUPPORTED_LOAD_BACKENDS = [TGISBackend.backend_type, backend_types.LOCAL]
    ## Module Interface ##

    def __init__(
        self,
        base_model_name,
        prompt_cache_id,
        eos_token,
        verbalizer,
        enable_backend=True,
        tgis_backend=None,
    ) -> None:
        super().__init__()
        # Configure the internal client
        # NOTE: This is made optional for the cases where we do not need to execute `.run` function
        # for example, bootstrapping a model to caikit format and saving.
        if enable_backend:
            # get_client will also launch a local TGIS process and get the model
            # loaded when using the local TGIS backend
            self._client = tgis_backend.get_client(base_model_name)

        self.base_model_name = base_model_name
        self._prompt_cache_id = prompt_cache_id
        self.eos_token = eos_token
        self.verbalizer = verbalizer
        self.enable_backend = enable_backend

    @classmethod
    def load(cls, model_path: str, load_backend: BackendBase) -> "PeftPromptTuningTGIS":
        """Load a TGIS Peft Prompt Tuning distributed module. Note that we do not
        leverage artifacts stored within the model here, and we assume that the
        prompt vector is already available at a place that the TGIS server can pick it
        up.

        Args:
            model_path: str
                Path to the model to be loaded.
            load_backend: BackendBase
                Backend object to be used to run inference with.
        Returns:
            PeftPromptTuningTGIS
                Instance of this class built from the on disk model.
        """
        error.type_check("<NLP85069377E>", TGISBackend, load_backend=load_backend)
        config = ModuleConfig.load(model_path)
        eos_token = config.eos_token
        verbalizer = config.verbalizer
        dir_name = os.path.split(model_path)[-1]
        # NOTE: base_model_name is used as "model_id" when calling to TGIS backend
        base_model_name = config.get("base_model_name", "")
        prompt_cache_id = dir_name
        error.type_check("<NLP24633932E>", str, prompt_cache_id=prompt_cache_id)
        # NOTE: prompt model config stores a base_model_config
        # which can be used to validate if the prompt is tuned
        # for the model it is being used with. However,
        # we are currently not accessing or assuming the accessibilty of
        # base model, thus not validating.
        # NOTE: When we access base_model_config, we need to make sure
        # we convert make it valid json compatible dict (aka doesn't have non string keys)
        log.debug("Prompt ID: %s", prompt_cache_id)
        log.debug("TGIS model ID: %s", base_model_name)
        return cls(
            base_model_name,
            prompt_cache_id,
            eos_token,
            verbalizer,
            tgis_backend=load_backend,
        )

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

    @TextGenerationTask.taskmethod()
    def run(self, text, preserve_input_text=False, max_new_tokens=20, min_new_tokens=0):
        """Run inference against the model running in TGIS. Currently we leverage greedy decoding
        and apply the same verbalizer used for training the local model prior to sending the
        request to TGIS.

        Args:
            text: str
                Source string to be encoded for generation.
            preserve_input_text: str
                Whether or not the source string should be contained in the generated output,
                e.g., as a prefix.
            max_new_tokens: int
                The maximum numbers of tokens to generate.
                Default: 20
            min_new_tokens: int
                The minimum numbers of tokens to generate.
                Default: 0 - means no minimum

        Returns:
            GeneratedTextResult
                Generated text result produced by TGIS.
        """
        error.value_check(
            "<NLP87360638E>",
            self.enable_backend,
            "Backend must be configured and loaded with this module before executing `run` call.",
        )
        verbalized_text = render_verbalizer(self.verbalizer, {"input": text})
        log.debug("Building protobuf request to send to TGIS")

        params = get_params(
            preserve_input_text=preserve_input_text,
            eos_token=self.eos_token,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
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
            "<NLP12333421E>",
            len(batch_response.responses) == 1,
            f"Got {len(batch_response.responses)} responses for a single request",
        )
        response = batch_response.responses[0]

        return GeneratedTextResult(
            generated_text=response.text,
            generated_tokens=response.generated_token_count,
            finish_reason=response.stop_reason,
            producer_id=self.PRODUCER_ID,
        )

    @TextGenerationTask.taskmethod(output_streaming=True)
    def run_stream_out(
        self, text: str, preserve_input_text=False, max_new_tokens=20, min_new_tokens=0
    ) -> Iterable[GeneratedTextStreamResult]:
        """Run output stream inferencing against the model running in TGIS

        Args:
            text: str
                Source string to be encoded for generation.
            preserve_input_text: str
                Whether or not the source string should be contained in the generated output,
                e.g., as a prefix.
            max_new_tokens: int
                The maximum numbers of tokens to generate.
                Default: 20
            min_new_tokens: int
                The minimum numbers of tokens to generate.
                Default: 0 - means no minimum

        Returns:
            Iterable[GeneratedTextStreamResult]
        """
        # pylint: disable=duplicate-code
        error.value_check(
            "<NLP62995899E>",
            self.enable_backend,
            "Backend must be configured and loaded with this module before executing `run` call.",
        )
        verbalized_text = render_verbalizer(self.verbalizer, {"input": text})
        log.debug("Building protobuf request to send to TGIS")

        params = get_params(
            preserve_input_text=preserve_input_text,
            eos_token=self.eos_token,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
        )

        gen_req = generation_pb2.GenerationRequest(text=verbalized_text)

        request = generation_pb2.SingleGenerationRequest(
            request=gen_req,
            model_id=self.base_model_name,
            prefix_id=self._prompt_cache_id,
            params=params,
        )

        # stream GenerationResponse
        stream_response = self._client.GenerateStream(request)

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
