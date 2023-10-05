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
from pathlib import Path
from typing import List
import os

# Third Party
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search, normalize_embeddings, dot_score

# First Party
from caikit.core import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.core.exceptions import error_handler
import alog

from caikit_nlp.data_model.reranker import RerankPrediction, RerankQueryResult, RerankScore, RerankDocuments
from .rerank_task import RerankTask

logger = alog.use_channel("<EMBD_BLK>")
error = error_handler.get(logger)

HOME = Path.home()
DEFAULT_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@module(
    "00110203-0405-0607-0809-0a0b02dd0e0f",
    "RerankerModule",
    "0.0.1",
    RerankTask,
)
class Rerank(ModuleBase):

    def __init__(
            self,
            model: SentenceTransformer,
    ):
        """Initialize
        This function gets called by `.load` and `.train` function
        which initializes this module.
        """
        super().__init__()
        self.model = model

    @classmethod
    def load(cls, model_path: str) -> "Rerank":
        """Load a model

        Args:
            model_path: str
                Path to the config dir of the model to be loaded.

        Returns:
            Rerank
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

    def run(self, queries: List[str], documents: RerankDocuments, top_n: int = 10) -> RerankPrediction:
        """Run inference on model.
        Args:
            queries: List[str]
            documents:  RerankDocuments
            top_n:  int
        Returns:
            RerankPrediction
        """

        if len(queries) < 1:
            return RerankPrediction()

        if len(documents.documents) < 1:
            return RerankPrediction()

        if top_n < 1:
            top_n = 10  # Default to 10 (instead of JSON default 0)

        # Using input document dicts so get "text" else "_text" else default to ""
        doc_texts = [srd.document.get("text") or srd.document.get("_text", "") for srd in documents.documents]

        doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)
        doc_embeddings = doc_embeddings.to(self.model.device)
        doc_embeddings = normalize_embeddings(doc_embeddings)

        query_embeddings = self.model.encode(queries, convert_to_tensor=True)
        query_embeddings = query_embeddings.to(self.model.device)
        query_embeddings = normalize_embeddings(query_embeddings)

        res = semantic_search(query_embeddings, doc_embeddings, top_k=top_n, score_function=dot_score)

        for r in res:
            for x in r:
                x['document'] = documents.documents[x['corpus_id']].document

        results = [RerankQueryResult([RerankScore(**x) for x in r]) for r in res]

        return RerankPrediction(results=results)

    @classmethod
    def bootstrap(cls, base_model_path: str) -> "Rerank":
        """Bootstrap a sentence-transformers model

        Args:
            base_model_path: str
                Path to the model to be loaded.
        """
        model = SentenceTransformer(
            base_model_path,
            cache_folder=f"{HOME}/.cache/huggingface/sentence_transformers",
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
