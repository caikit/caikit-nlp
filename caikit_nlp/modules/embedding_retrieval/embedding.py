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

import alog
from caikit.core import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.core.exceptions import error_handler
from caikit_nlp.data_model.embedding_vectors import EmbeddingResult, Vector1D

from sentence_transformers import SentenceTransformer

from .embedding_retrieval_task import EmbeddingRetrievalTask

import os
from pathlib import Path
from typing import List

logger = alog.use_channel("<EMBD_BLK>")
error = error_handler.get(logger)

HOME = Path.home()
DEFAULT_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@module(
    "EEB12558-B4FA-4F34-A9FD-3F5890E9CD3F",
    "EmbeddingModule",
    "0.0.1",
    EmbeddingRetrievalTask,
)
class EmbeddingModule(ModuleBase):

    def __init__(
        self,
        model: SentenceTransformer,
    ):
        super().__init__()
        self.model = model


    @classmethod
    def load(cls, model_path: str) -> "EmbeddingModule":
        """Load a sequence classification model

        Args:
            model_path: str
                Path to the config dir of the model to be loaded.

        Returns:
            EmbeddingModule
                Instance of this class built from the model.
        """

        config = ModuleConfig.load(model_path)
        load_path = config.get("model_artifacts")

        artifact_path = False
        if load_path:
            if os.path.isabs(load_path) and os.path.isdir(load_path):
                artifact_path = load_path
            else:
                full_path = os.path.join(model_path, load_path)
                if os.path.isdir(full_path):
                    artifact_path = full_path

        if not artifact_path:
            artifact_path = config.get("hf_model", DEFAULT_HF_MODEL)

        return cls.bootstrap(artifact_path)

    def run(
            self, input: List[str], **kwargs   # pylint: disable=redefined-builtin
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
            vectors.append(Vector1D(vector))

        return EmbeddingResult(vectors)

    @classmethod
    def bootstrap(cls, base_model_path: str) -> "EmbeddingModule":
        """Bootstrap a HuggingFace transformer-based sequence classification model

        Args:
            base_model_path: str
                Path to the model to be loaded.
        """
        model = SentenceTransformer(
            base_model_path,
            cache_folder=f"{HOME}/.cache/huggingface/sentence_transformers"
        )
        return cls(
            model=model,
        )

    def save(self, model_path: str):
        """Save model in target path

        Args:
            model_path: str
                Path to store model artifact(s)
        """
        saver = ModuleSaver(
            self,
            model_path=model_path,
        )

        # Extract object to be saved
        with saver:
            saver.update_config({"model_artifacts": "."})
            if self.model:  # This condition allows for empty placeholders
                self.model.save(model_path)
