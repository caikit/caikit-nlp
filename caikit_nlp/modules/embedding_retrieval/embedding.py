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
from typing import List
import os

# Third Party
from sentence_transformers import SentenceTransformer
import numpy as np

# First Party
from caikit.core import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.core.exceptions import error_handler
import alog

# Local
from .embedding_retrieval_task import EmbeddingRetrievalTask
from caikit_nlp.data_model.embedding_vectors import (
    EmbeddingResult,
    NpFloat32Sequence,
    NpFloat64Sequence,
    PyFloatSequence,
    Vector1D,
)

logger = alog.use_channel("<EMBD_BLK>")
error = error_handler.get(logger)


@module(
    "EEB12558-B4FA-4F34-A9FD-3F5890E9CD3F",
    "EmbeddingModule",
    "0.0.1",
    EmbeddingRetrievalTask,
)
class EmbeddingModule(ModuleBase):

    _ARTIFACTS_PATH_KEY = "artifacts_path"
    _ARTIFACTS_PATH_DEFAULT = "artifacts"
    _HF_HUB_KEY = "hf_model"

    def __init__(
        self,
        model: SentenceTransformer,
    ):
        super().__init__()
        self.model = model

    @classmethod
    def load(cls, model_path: str) -> "EmbeddingModule":
        """Load model

        Args:
            model_path: str
                Path to the config dir of the model to be loaded.

        Returns:
            EmbeddingModule
                Instance of this class built from the model.
        """

        config = ModuleConfig.load(model_path)

        artifacts_path = config.get(cls._ARTIFACTS_PATH_KEY)
        if artifacts_path:
            model_name_or_path = os.path.abspath(
                os.path.join(model_path, artifacts_path)
            )
            error.dir_check("<NLP34197772E>", model_name_or_path)
        else:
            # If no artifacts_path, look for hf_model Hugging Face model by name (or path)
            model_name_or_path = config.get(cls._HF_HUB_KEY)
            error.value_check(
                "<NLP07391618E>",
                model_name_or_path,
                ValueError(
                    f"Model config missing '{cls._ARTIFACTS_PATH_KEY}' or '{cls._HF_HUB_KEY}'"
                ),
            )

        return cls.bootstrap(model_name_or_path=model_name_or_path)

    def run(
        self, input: List[str], **kwargs  # pylint: disable=redefined-builtin
    ) -> EmbeddingResult:
        """Run inference on model.
        Args:
            input: List[str]
                Input text to be processed
        Returns:
            EmbeddingResult: the output
        """

        sentences: List[str]
        if isinstance(input, str):
            sentences = [input]
        else:
            sentences = input

        result = self.model.encode(sentences)

        vectors: List[Vector1D] = []
        for vector in result:
            if vector.dtype == np.float32:
                vectors.append(Vector1D(NpFloat32Sequence(vector)))
            elif vector.dtype == np.float64:
                vectors.append(Vector1D(NpFloat64Sequence(vector)))
            else:
                vectors.append(Vector1D(PyFloatSequence(vector)))

        return EmbeddingResult(vectors)

    @classmethod
    def bootstrap(cls, model_name_or_path: str) -> "EmbeddingModule":
        """Bootstrap a sentence-transformers model

        Args:
            model_name_or_path: str
                Model name (Hugging Face hub) or path to model to load.
        """
        return cls(model=SentenceTransformer(model_name_or_path=model_name_or_path))

    def save(self, model_path: str, *args, **kwargs):
        """Save model using config in model_path

        Args:
            model_path: str
                Path to model config
        """

        error.type_check("<NLP82314992E>", str, model_path=model_path)
        error.value_check(
            "<NLP40145207E>",
            model_path is not None and model_path.strip(),
            f"model_path '{model_path}' is invalid",
        )

        model_path = os.path.abspath(
            model_path.strip()
        )  # No leading/trailing spaces sneaky weirdness

        if os.path.exists(model_path):
            error(
                "<NLP44708517E>",
                FileExistsError(f"model_path '{model_path}' already exists"),
            )

        saver = ModuleSaver(
            module=self,
            model_path=model_path,
        )

        # Save update config (artifacts_path) and save artifacts
        with saver:
            artifacts_path = saver.config.get(self._ARTIFACTS_PATH_KEY)
            if not artifacts_path:
                artifacts_path = self._ARTIFACTS_PATH_DEFAULT
                saver.update_config({self._ARTIFACTS_PATH_KEY: artifacts_path})
            if self.model:  # This condition allows for empty placeholders
                artifacts_path = os.path.abspath(
                    os.path.join(model_path, artifacts_path)
                )
                self.model.save(artifacts_path, create_model_card=True)
