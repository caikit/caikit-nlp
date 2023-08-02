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
from typing import Iterable, Optional
import os

# First Party
from caikit.core.module_backends import BackendBase, backend_types
from caikit.core.modules import module, ModuleBase, ModuleConfig, ModuleSaver
from caikit.core.toolkit import error_handler
from caikit.interfaces.nlp.data_model import (
    GeneratedTextResult,
    GeneratedTextStreamResult,
)
from caikit.interfaces.nlp.tasks import TextGenerationTask
from caikit_tgis_backend import TGISBackend
import alog

# Local
from ...resources.pretrained_model import (
    HFAutoCausalLM,
    HFAutoSeq2SeqLM,
    PretrainedModelBase,
)
from .text_generation_local import TextGeneration
from ...toolkit.tgis_utils import TGISGenerationClient

log = alog.use_channel("TXT_GEN")
error = error_handler.get(log)


@module(backend_type=TGISBackend.backend_type, base_module=TextGeneration)
class TextGenerationTGIS(ModuleBase):
    """Module to provide text generation capabilities"""

    SUPPORTED_LOAD_BACKENDS = [TGISBackend.backend_type, backend_types.LOCAL]

    supported_resources = [HFAutoCausalLM, HFAutoSeq2SeqLM]

    def __init__(
        self,
        base_model_name: str,
        base_model: Optional[PretrainedModelBase] = None,
        bos_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        tgis_backend: Optional[TGISBackend] = None,
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
        self._client = None
        if tgis_backend:
            self._client = tgis_backend.get_client(base_model_name)
            # mark that the model is loaded so that we can unload it later
            self._model_loaded = True

        self._bos_token = bos_token
        self._sep_token = sep_token
        self._eos_token = eos_token
        self._pad_token = pad_token
        self.tgis_generation_client = TGISGenerationClient(
            self.base_model_name, self._eos_token, self._client, self.PRODUCER_ID
        )

    def __del__(self):
        # nothing to unload if we didn't finish loading
        if self._model_loaded and self.load_backend:
            self.load_backend.unload_model(self._model_path)

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
        saver = ModuleSaver(self, model_path=model_path)
        with saver:
            saver.update_config(
                {
                    "base_model_name": self.base_model_name,
                    "bos_token": self._bos_token,
                    "sep_token": self._sep_token,
                    "eos_token": self._eos_token,
                    "pad_token": self._pad_token,
                }
            )
            if self.base_model:
                artifacts_dir = "artifacts"
                log.debug("Saving model artifacts to %s", artifacts_dir)
                saver.update_config({"artifact_path": artifacts_dir})
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
        artifacts_path = config.artifact_path
        if artifacts_path:
            base_model_name = os.path.join(model_path, artifacts_path)
            error.dir_check("<NLP01983374E>", base_model_name)
            log.debug("Loading with on-disk artifacts: %s", base_model_name)
        else:
            base_model_name = config.base_model_name
            error.type_check("<NLP90686335E>", str, base_model_name=base_model_name)
            log.debug("Loading with model name: %s", base_model_name)
        return cls(
            base_model_name,
            bos_token=config.bos_token,
            sep_token=config.sep_token,
            eos_token=config.eos_token,
            pad_token=config.pad_token,
            tgis_backend=load_backend,
        )

    @TextGenerationTask.taskmethod()
    def run(
        self, text, preserve_input_text=False, max_new_tokens=20, min_new_tokens=0
    ) -> GeneratedTextResult:
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
            GeneratedTextResult
                Generated text result produced by TGIS.
        """
        if self._model_loaded:
            return self.tgis_generation_client.unary_generate(
                text, preserve_input_text, max_new_tokens, min_new_tokens
            )

    @TextGenerationTask.taskmethod(output_streaming=True)
    def run_stream_out(
        self, text: str, preserve_input_text=False, max_new_tokens=20, min_new_tokens=0
    ) -> Iterable[GeneratedTextStreamResult]:
        """Run output stream inferencing for text generation module.

        Args:
            text: str
                Source string to be encoded for generation.
            preserve_input_text: bool
                Whether or not the source string should be contained in the generated output,
                e.g., as a prefix.
            max_new_tokens: int
                Maximum tokens for the model to generate
            min_new_tokens: int
                Minimum tokens for the model to generate

        Returns:
            Iterable[GeneratedTextStreamResult]
        """
        if self._model_loaded:
            return self.tgis_generation_client.stream_generate(
                text, preserve_input_text, max_new_tokens, min_new_tokens
            )
