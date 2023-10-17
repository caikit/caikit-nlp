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
from sentence_transformers import SentenceTransformer
import numpy as np

# First Party
from caikit.core import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.core.exceptions import error_handler
import alog

# Local
from .embedding_retrieval_task import EmbeddingRetrievalTask
from caikit_nlp.data_model.embedding_vectors import (
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

    def __init__(
        self,
        model: SentenceTransformer,
    ):
        super().__init__()
        self.model = model

    @classmethod
    def load(cls, model_path: str, *args, **kwargs) -> "EmbeddingModule":
        """Load model

        Args:
            model_path: str
                Path to the config dir under the model_id (where the config.yml lives)

        Returns:
            EmbeddingModule
                Instance of this class built from the model.
        """

        config = ModuleConfig.load(model_path)
        artifacts_path = config.get(cls._ARTIFACTS_PATH_KEY)

        error.value_check(
            "<NLP07391618E>",
            artifacts_path,
            ValueError(f"Model config missing '{cls._ARTIFACTS_PATH_KEY}'"),
        )

        artifacts_path = os.path.abspath(os.path.join(model_path, artifacts_path))
        error.dir_check("<NLP34197772E>", artifacts_path)

        return cls.bootstrap(model_name_or_path=artifacts_path)

    def run(
        self, input: str, **kwargs  # pylint: disable=redefined-builtin
    ) -> Vector1D:
        """Run inference on model.
        Args:
            input: str
                Input text to be processed
        Returns:
            Vector1D: the output
        """

        error.type_check("<NLP27491611E>", str, input=input)

        embeddings = self.model.encode(input)

        if embeddings.dtype == np.float32:
            data = NpFloat32Sequence(embeddings)
        elif embeddings.dtype == np.float64:
            data = NpFloat64Sequence(embeddings)
        else:
            data = PyFloatSequence(embeddings)

        return Vector1D(data)

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

        model_config_path = model_path  # because the param name is misleading

        error.type_check("<NLP82314992E>", str, model_path=model_config_path)
        error.value_check(
            "<NLP40145207E>",
            model_config_path is not None and model_config_path.strip(),
            f"model_path '{model_config_path}' is invalid",
        )

        model_config_path = os.path.abspath(
            model_config_path.strip()
        )  # No leading/trailing spaces sneaky weirdness

        os.makedirs(model_config_path, exist_ok=False)
        saver = ModuleSaver(
            module=self,
            model_path=model_config_path,
        )

        # Get and update config (artifacts_path)
        artifacts_path = saver.config.get(self._ARTIFACTS_PATH_KEY)
        if not artifacts_path:
            artifacts_path = self._ARTIFACTS_PATH_DEFAULT
            saver.update_config({self._ARTIFACTS_PATH_KEY: artifacts_path})

        # Save the model
        artifacts_path = os.path.abspath(
            os.path.join(model_config_path, artifacts_path)
        )
        self.model.save(artifacts_path, create_model_card=True)

        # Save the config
        ModuleConfig(saver.config).save(model_config_path)
