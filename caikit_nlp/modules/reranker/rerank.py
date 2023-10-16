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
from typing import List, Optional
import os

# Third Party
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import dot_score, normalize_embeddings, semantic_search

# First Party
from caikit.core import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.core.data_model.json_dict import JsonDict
from caikit.core.exceptions import error_handler
import alog

# Local
from .rerank_task import RerankTask
from caikit_nlp.data_model.reranker import (
    RerankPrediction,
    RerankQueryResult,
    RerankScore,
)

logger = alog.use_channel("<EMBD_BLK>")
error = error_handler.get(logger)


@module(
    "00110203-0405-0607-0809-0a0b02dd0e0f",
    "RerankerModule",
    "0.0.1",
    RerankTask,
)
class Rerank(ModuleBase):

    _ARTIFACTS_PATH_KEY = "artifacts_path"
    _ARTIFACTS_PATH_DEFAULT = "artifacts"
    _HF_HUB_KEY = "hf_model"

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

        artifacts_path = config.get(cls._ARTIFACTS_PATH_KEY)
        if artifacts_path:
            model_name_or_path = os.path.abspath(
                os.path.join(model_path, artifacts_path)
            )
            error.dir_check("<NLP01027299E>", model_name_or_path)
        else:
            # If no artifacts_path, look for hf_model Hugging Face model by name (or path)
            model_name_or_path = config.get(cls._HF_HUB_KEY)
            error.value_check(
                "<NLP22444208E>",
                model_name_or_path,
                ValueError(
                    f"Model config missing '{cls._ARTIFACTS_PATH_KEY}' or '{cls._HF_HUB_KEY}'"
                ),
            )

        return cls.bootstrap(model_name_or_path=model_name_or_path)

    def run(
        self,
        queries: List[str],
        documents: List[JsonDict],
        top_n: Optional[int] = None,
    ) -> RerankPrediction:
        """Run inference on model.
        Args:
            queries: List[str]
            documents:  List[JsonDict]
            top_n:  Optional[int]
        Returns:
            RerankPrediction
        """

        error.type_check(
            "<NLP09038249E>",
            list,
            queries=queries,
            documents=documents,
        )

        if len(queries) < 1 or len(documents) < 1:
            return RerankPrediction([])

        if top_n is None or top_n < 1:
            top_n = len(documents)

        # Using input document dicts so get "text" else "_text" else default to ""
        doc_texts = [srd.get("text") or srd.get("_text", "") for srd in documents]

        doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)
        doc_embeddings = doc_embeddings.to(self.model.device)
        doc_embeddings = normalize_embeddings(doc_embeddings)

        query_embeddings = self.model.encode(queries, convert_to_tensor=True)
        query_embeddings = query_embeddings.to(self.model.device)
        query_embeddings = normalize_embeddings(query_embeddings)

        res = semantic_search(
            query_embeddings, doc_embeddings, top_k=top_n, score_function=dot_score
        )

        for r in res:
            for x in r:
                x["document"] = documents[x["corpus_id"]]

        results = [RerankQueryResult([RerankScore(**x) for x in r]) for r in res]

        return RerankPrediction(results=results)

    @classmethod
    def bootstrap(cls, model_name_or_path: str) -> "Rerank":
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
