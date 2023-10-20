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
from sentence_transformers.util import (
    cos_sim,
    dot_score,
    normalize_embeddings,
    semantic_search,
)

# First Party
from caikit.core import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.core.data_model.json_dict import JsonDict
from caikit.core.exceptions import error_handler
import alog

# Local
from .embedding_retrieval_task import EmbeddingTask, EmbeddingTasks
from .rerank_task import RerankTask, RerankTasks
from .sentence_similarity_task import SentenceSimilarityTask, SentenceSimilarityTasks
from caikit_nlp.data_model import (
    ListOfVector1D,
    RerankPrediction,
    RerankQueryResult,
    RerankScore,
    SentenceListScores,
    SentenceScores,
    Vector1D,
)

logger = alog.use_channel("<EMBD_BLK>")
error = error_handler.get(logger)


@module(
    "EEB12558-B4FA-4F34-A9FD-3F5890E9CD3F",
    "Text Embedding",
    "0.0.1",
    tasks=[
        EmbeddingTask,
        EmbeddingTasks,
        SentenceSimilarityTask,
        SentenceSimilarityTasks,
        RerankTask,
        RerankTasks,
    ],
)
class TextEmbedding(ModuleBase):

    _ARTIFACTS_PATH_KEY = "artifacts_path"
    _ARTIFACTS_PATH_DEFAULT = "artifacts"

    def __init__(
        self,
        model: SentenceTransformer,
    ):
        super().__init__()
        self.model = model

    @classmethod
    def load(cls, model_path: str, *args, **kwargs) -> "TextEmbedding":
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

    @EmbeddingTask.taskmethod()
    def run_embedding(self, text: str) -> Vector1D:  # pylint: disable=redefined-builtin
        """Get embedding for a string.
        Args:
            text: str
                Input text to be processed
        Returns:
            Vector1D: the output
        """
        error.type_check("<NLP27491611E>", str, text=text)

        return Vector1D.from_embeddings(self.model.encode(text))

    @EmbeddingTasks.taskmethod()
    def run_embeddings(
        self, texts: List[str]  # pylint: disable=redefined-builtin
    ) -> ListOfVector1D:
        """Run inference on model.
        Args:
            texts: List[str]
                List of input texts to be processed
        Returns:
            List[Vector1D]: List vectors. One for each input text (in order).
             Each vector is a list of floats (supports various float types).
        """
        if isinstance(
            texts, str
        ):  # encode allows str, but the result would lack a dimension
            texts = [texts]

        embeddings = self.model.encode(texts)
        results = [Vector1D.from_embeddings(e) for e in embeddings]
        return ListOfVector1D(results=results)

    @SentenceSimilarityTask.taskmethod()
    def run_sentence_similarity(
        self, source_sentence: str, sentences: List[str]
    ) -> SentenceScores:  # pylint: disable=arguments-differ
        """Run inference on model.
        Args:
            source_sentence: str
            sentences: List[str]
                Sentences to compare to source_sentence
        Returns:
            SentenceScores
        """

        source_embedding = self.model.encode(source_sentence)
        embeddings = self.model.encode(sentences)

        res = cos_sim(source_embedding, embeddings)
        return SentenceScores(res.tolist()[0])

    @SentenceSimilarityTasks.taskmethod()
    def run_sentence_similarities(
        self, source_sentences: List[str], sentences: List[str]
    ) -> SentenceListScores:  # pylint: disable=arguments-differ
        """Run inference on model.
        Args:
            source_sentences: List[str]
            sentences: List[str]
                Sentences to compare to source_sentences
        Returns:
            SentenceListScores Similarity scores for each source sentence in order.
                each SentenceScores contains the source-sentence's score for each sentence in order.
        """

        source_embedding = self.model.encode(source_sentences)
        embeddings = self.model.encode(sentences)

        res = cos_sim(source_embedding, embeddings)
        float_list_list = res.tolist()
        return SentenceListScores(
            results=[SentenceScores(fl) for fl in float_list_list]
        )

    @RerankTask.taskmethod()
    def run_rerank_query(
        self,
        query: str,
        documents: List[JsonDict],
        top_n: Optional[int] = None,
        return_documents: bool = True,
        return_query: bool = True,
        return_text: bool = True,
    ) -> RerankQueryResult:
        """Run inference on model.
        Args:
            query: str
            documents:  List[JsonDict]
            top_n:  Optional[int]
        Returns:
            RerankQueryResult
        """

        error.type_check(
            "<NLP05323654E>",
            str,
            query=query,
        )

        results = self.run_rerank_queries(
            queries=[query],
            documents=documents,
            top_n=top_n,
            return_documents=return_documents,
            return_queries=return_query,
            return_text=return_text,
        ).results

        return (
            results[0]
            if len(results) > 0
            else RerankQueryResult(scores=[], query=query if return_query else None)
        )

    @RerankTasks.taskmethod()
    def run_rerank_queries(
        self,
        queries: List[str],
        documents: List[JsonDict],
        top_n: Optional[int] = None,
        return_documents: bool = True,
        return_queries: bool = True,
        return_text: bool = True,
    ) -> RerankPrediction:
        """Run inference on model.
        Args:
            queries: List[str]
            documents:  List[JsonDict]
            top_n:  Optional[int]
            return_documents:  bool
            return_queries:  bool
            return_text:  bool
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
        def get_text(srd):
            return srd.get("text") or srd.get("_text", "")

        doc_texts = [get_text(srd) for srd in documents]

        doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)
        doc_embeddings = doc_embeddings.to(self.model.device)
        doc_embeddings = normalize_embeddings(doc_embeddings)

        query_embeddings = self.model.encode(queries, convert_to_tensor=True)
        query_embeddings = query_embeddings.to(self.model.device)
        query_embeddings = normalize_embeddings(query_embeddings)

        res = semantic_search(
            query_embeddings, doc_embeddings, top_k=top_n, score_function=dot_score
        )

        # Fixup result dicts
        for r in res:
            for x in r:
                # Renaming corpus_id to index
                corpus_id = x.pop("corpus_id")
                x["index"] = corpus_id
                # Optionally adding the original document and/or just the text that was used
                if return_documents:
                    x["document"] = documents[corpus_id]
                if return_text:
                    x["text"] = get_text(documents[corpus_id])

        def add_query(q):
            return queries[q] if return_queries else None

        results = [
            RerankQueryResult(query=add_query(q), scores=[RerankScore(**x) for x in r])
            for q, r in enumerate(res)
        ]

        return RerankPrediction(results=results)

    @classmethod
    def bootstrap(cls, model_name_or_path: str) -> "TextEmbedding":
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
