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

# First Party
from caikit.core import modules
from caikit.core.module_backends import BackendBase, backend_types
from caikit.core.modules import ModuleBase, ModuleConfig, ModuleSaver
from caikit.core.toolkit import error_handler
from caikit_tgis_backend import TGISBackend
from caikit_tgis_backend.protobufs import generation_pb2
import alog

# Local
from ...data_model import GeneratedResult
from ...resources.pretrained_model import (
    HFAutoCausalLM,
    HFAutoSeq2SeqLM,
    PretrainedModelBase,
)
from .text_generation_local import TextGeneration

log = alog.use_channel("TXT_GEN")
error = error_handler.get(log)


@modules.module(backend_type=TGISBackend.backend_type, base_module=TextGeneration)
class TextGenerationTGIS(ModuleBase):
    """Module to provide text generation capabilities"""

    SUPPORTED_LOAD_BACKENDS = [TGISBackend.backend_type, backend_types.LOCAL]

    supported_resources = [HFAutoCausalLM, HFAutoSeq2SeqLM]

    def __init__(
        self,
        base_model_name: str,
        base_model: PretrainedModelBase = None,
        bos_token: str = None,
        sep_token: str = None,
        eos_token: str = None,
        pad_token: str = None,
        tgis_backend: TGISBackend = None,
    ):
        super().__init__()

        error.type_check("<NLP00609194E>", str, allow_none=True, bos_token=bos_token)
        error.type_check("<NLP72469403E>", str, allow_none=True, sep_token=sep_token)
        error.type_check("<NLP48137045E>", str, allow_none=True, eos_token=eos_token)
        error.type_check("<NLP53511308E>", str, allow_none=True, pad_token=pad_token)
        self.base_model = base_model
        self.base_model_name = base_model_name

        # Set _model_loaded as False by default. This will only get set to True if
        # we enable the tgis_backend and we are able to fetch the client successfully.
        self._model_loaded = False
        # Configure the internal client
        # NOTE: This is made optional for the cases where we do not need to execute `.run` function
        # for example, bootstrapping a model to caikit format and saving.
        if tgis_backend:
            self._client = tgis_backend.get_client(base_model_name)
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
    def bootstrap(cls, base_model_path: str, load_backend: BackendBase = None):
        """Function to bootstrap a pre-trained transformers model and
        get a caikit text-generation 'model'.

        Args:
            base_model_path: str
                Path to transformers model
                NOTE: Model path needs to contain tokenizer as well
            load_backend: BackendBase
                Backend object to be used to run inference with.
                NOTE: this is required for inferencing. It is
                made optional to support the model conversion use-case
        Returns:
            caikit_nlp.blocks.text_generation.TextGeneration
                Object of TextGeneration class (model)
        """
        text_generation_inst = TextGeneration.bootstrap(base_model_path)
        bos_token = text_generation_inst.base_model._tokenizer.bos_token
        sep_token = text_generation_inst.base_model._tokenizer.sep_token
        eos_token = text_generation_inst.base_model._tokenizer.eos_token or None
        pad_token = text_generation_inst.base_model._tokenizer.pad_token
        return cls(
            text_generation_inst.base_model_name,
            text_generation_inst.base_model,
            bos_token=bos_token,
            sep_token=sep_token,
            eos_token=eos_token,
            pad_token=pad_token,
            tgis_backend=load_backend,
        )

    def save(self, model_path):
        """Save caikit model

        Args:
            model_path: str
                Folder to save text-generation caikit model
        """
        saver = ModuleSaver(
            self,
            model_path=model_path,
        )
        with saver:
            artifacts_dir = "artifacts"
            saver.update_config(
                {
                    "artifact_path": artifacts_dir,
                    "bos_token": self._bos_token,
                    "sep_token": self._sep_token,
                    "eos_token": self._eos_token,
                    "pad_token": self._pad_token,
                }
            )
            if self.base_model:
                # This will save both tokenizer and base model
                self.base_model.save(
                    model_path,
                    tokenizer_dirname=artifacts_dir,
                    base_model_dirname=artifacts_dir,
                )

    @classmethod
    def load(cls, model_path: str, load_backend: BackendBase) -> "TextGeneration":
        """Function to load text-generation model

        Args:
            model_path: str
                Path to the model to be loaded.
            load_backend: BackendBase
                Backend object to be used to run inference with.
        Returns:
            TextGeneration
                Instance of this class built from the on disk model.
        """
        error.type_check("<NLP03521359E>", TGISBackend, load_backend=load_backend)

        config = ModuleConfig.load(model_path)
        base_model_path = config.get("artifact_path", "")
        base_model_path = os.path.join(model_path, base_model_path)
        error.dir_check("<NLP01983374E>", base_model_path)
        return cls(
            base_model_path,
            bos_token=config.bos_token,
            sep_token=config.sep_token,
            eos_token=config.eos_token,
            pad_token=config.pad_token,
            tgis_backend=load_backend,
        )

    def run(self, text, preserve_input_text=False, max_new_tokens=20, min_new_tokens=0):
        """Run inference against the model running in TGIS.

        Args:
            text: str
                Source string to be encoded for generation.
            preserve_input_text: bool
                Whether or not the source string should be contained in the generated output,
                e.g., as a prefix.
            max_new_tokens: int
                The maximum numbers of tokens to generate.
                Default: 20
            min_new_tokens: int
                The minimum numbers of tokens to generate.
                Default: 0 - means no minimum
        Returns:
            GeneratedResult
                Generated text result produced by TGIS.
        """
        log.debug("Building protobuf request to send to TGIS")
        # pylint: disable=duplicate-code
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
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
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

            # pylint: disable=duplicate-code
            error.value_check(
                "<NLP38899018E>",
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
