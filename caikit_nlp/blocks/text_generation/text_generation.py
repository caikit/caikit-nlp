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
from typing import Optional, Union

# Third Party
from transformers import AutoConfig

# First Party
from caikit_tgis_backend import TGISBackend
from caikit_tgis_backend.protobufs import generation_pb2
from caikit.core import BlockBase, block
from caikit.core.data_model import DataStream
from caikit.core import module_backend_config
from caikit.core.module_backends import backend_types
from caikit.core.module import ModuleConfig, ModuleSaver
from caikit.core.toolkit import error_handler
import alog

# Local
from ...data_model import GeneratedResult
from ...resources.pretrained_model import (
    HFAutoCausalLM,
    HFAutoSeq2SeqLM,
)

log = alog.use_channel("TXT_GEN")
error = error_handler.get(log)


@block(
    id="f9181353-4ccf-4572-bd1e-f12bcda26792",
    name="Text Generation",
    version="0.1.0",
    backend_type=TGISBackend.backend_type,
)
class TextGeneration(BlockBase):
    """Module to provide text generation capabilities"""

    supported_resources = [HFAutoCausalLM, HFAutoSeq2SeqLM]

    def __init__(
        self,
        base_model_name,
        base_model=None,
        bos_token=None,
        sep_token=None,
        eos_token=None,
        pad_token=None,
        enable_backend=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        error.type_check("<FPT00609194E>", str, allow_none=True, bos_token=bos_token)
        error.type_check("<FPT72469403E>", str, allow_none=True, sep_token=sep_token)
        error.type_check("<FPT48137045E>", str, allow_none=True, eos_token=eos_token)
        error.type_check("<FPT53511308E>", str, allow_none=True, pad_token=pad_token)
        self.base_model = base_model
        self.base_model_name = base_model_name
        # Configure the internal client
        if enable_backend:
            self._client = module_backend_config.get_backend(
                TGISBackend.backend_type
            ).get_client(base_model_name)
            # mark that the model is loaded so that we can unload it later
            self._model_loaded = True

        self._bos_token = bos_token
        self._sep_token = sep_token
        self._eos_token = eos_token
        self._pad_token = pad_token

    def __del__(self):
        # nothing to unload if we didn't finish loading
        if self._model_loaded:
            self.get_backend().unload_model(self._model_path)

    @classmethod
    def bootstrap(cls, base_model_path: str, enable_backend=False):
        """Function to bootstrap a pre-trained transformers model and
        get a caikit text-generation 'model'.

        Args:
            base_model_path: str
                Path to transformers model
            enable_backend: bool
                Enable loading the model in shared backend.
                # NOTE: this is required for inferencing. It is
                made optional just in provide support for model conversion use-case

        Returns:
            caikit_nlp.blocks.text_generation.TextGeneration
                Object of TextGeneration class (model)
        """
        model_config = AutoConfig.from_pretrained(base_model_path)

        resource_type = None
        for resource in cls.supported_resources:
            if model_config.model_type in resource.SUPPORTED_MODEL_TYPES:
                resource_type = resource
                break

        if not resource_type:
            error(
                "<FPT61784225E>",
                "{} model type is not supported currently!".format(
                    model_config.model_type
                ),
            )
        log.debug("Bootstrapping base resource [%s]", base_model_path)
        base_model = resource_type.bootstrap(base_model_path)
        bos_token = base_model._tokenizer.bos_token
        sep_token = base_model._tokenizer.sep_token
        eos_token = base_model._tokenizer.eos_token or None
        pad_token = base_model._tokenizer.pad_token
        return cls(
            base_model_path,
            base_model,
            bos_token=bos_token,
            sep_token=sep_token,
            eos_token=eos_token,
            pad_token=pad_token,
            enable_backend=enable_backend,
        )

    def save(self, artifact_path):
        """Save caikit model

        Args:
            artifact_path: str
                Folder to save text-generation caikit model
        """
        saver = ModuleSaver(
            self,
            model_path=artifact_path,
        )
        with saver:
            artifacts_dir = "artifacts"
            saver.update_config(
                {
                    "artifact_path": artifacts_dir,
                    "bos_token": self._bos_token,
                    "sep_token": self._sep_token,
                    "eos_token": self._eos_token,
                    # "truncate_input_tokens": self._truncate_input_tokens,
                    # "stop_sequences": self._stop_sequences,
                }
            )
            if self.base_model:
                # This will save both tokenizer and base model
                self.base_model.save(
                    artifact_path,
                    tok_dirname=artifacts_dir,
                    model_dirname=artifacts_dir,
                )

    @classmethod
    def load(cls, model_path: str) -> "TextGeneration":
        """Function to load text-generation model

        Args:
            model_path: str
                Path to the model to be loaded.
        Returns:
            TextGeneration
                Instance of this class built from the on disk model.
        """

        config = ModuleConfig.load(model_path)
        base_model_path = config.get("artifact_path", "")
        base_model_path = os.path.join(model_path, base_model_path)
        error.dir_check("<DWC20623231E>", base_model_path)
        return cls(
            base_model_path,
            bos_token=config.bos_token,
            sep_token=config.sep_token,
            eos_token=config.eos_token,
        )

    def run(self, text, preserve_input_text=False):
        """Run inference against the model running in TGIS.

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
        log.debug("Building protobuf request to send to TGIS")
        if self._model_loaded:
            res_options = generation_pb2.ResponseOptions(
                input_text=preserve_input_text,
                generated_tokens=True,
                input_tokens=False,
                token_logprobs=True,
                token_ranks=True,
            )
            stopping = generation_pb2.StoppingCriteria(
                stop_sequences=[self._eos_token],
            )
            params = generation_pb2.Parameters(
                response=res_options,
                stopping=stopping,
            )

            gen_reqs = [generation_pb2.GenerationRequest(text=text)]
            request = generation_pb2.BatchedGenerationRequest(
                requests=gen_reqs,
                model_id=self.base_model_name,
                params=params,
            )

            # Currently, we send a batch request of len(x)==1, so we expect one response back
            with alog.ContextTimer(log.trace, "TGIS request duration: "):
                batch_response = self._client.Generate(request)

            error.value_check(
                "<FPT45587981E>",
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