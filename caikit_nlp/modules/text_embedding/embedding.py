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
from collections.abc import Sized
from copy import deepcopy
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
)
import importlib
import os
import threading
import time

# Third Party
from torch import nn
from torch.backends import mps
from transformers import BatchEncoding
import numpy as np
import torch

# First Party
from caikit import get_config
from caikit.core import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.core.data_model.json_dict import JsonDict
from caikit.core.exceptions import error_handler
from caikit.interfaces.common.data_model.vectors import ListOfVector1D, Vector1D
from caikit.interfaces.nlp.data_model import (
    EmbeddingResult,
    EmbeddingResults,
    RerankResult,
    RerankResults,
    RerankScore,
    RerankScores,
    SentenceSimilarityResult,
    SentenceSimilarityResults,
    SentenceSimilarityScores,
    Token,
    TokenizationResults,
)
from caikit.interfaces.nlp.tasks import (
    EmbeddingTask,
    EmbeddingTasks,
    RerankTask,
    RerankTasks,
    SentenceSimilarityTask,
    SentenceSimilarityTasks,
    TokenizationTask,
)
import alog

# Local
from caikit_nlp.modules.text_embedding.utils import env_val_to_bool

logger = alog.use_channel("TXT_EMB")
error = error_handler.get(logger)


# To avoid dependency problems, make sentence-transformers an optional import and
# defer any ModuleNotFoundError until someone actually tries to init a model with this module.
try:
    sentence_transformers = importlib.import_module("sentence_transformers")
    # Third Party
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.model_card import SentenceTransformerModelCardData
    from sentence_transformers.similarity_functions import SimilarityFunction
    from sentence_transformers.util import batch_to_device, cos_sim, dot_score
    from sentence_transformers.util import (
        normalize_embeddings as normalize,  # avoid parameter shadowing
    )
    from sentence_transformers.util import semantic_search
except ModuleNotFoundError:
    # When it is not available, create a dummy that raises an error on attempted init()
    class SentenceTransformerNotAvailable:
        def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
            # Will reproduce the ModuleNotFoundError if/when anyone actually tries this module/model
            importlib.import_module("sentence_transformers")

    SentenceTransformer = SentenceTransformerNotAvailable

RT = TypeVar("RT")  # return type


class EmbeddingResultTuple(NamedTuple):
    """Output of SentenceTransformerWithTruncate.encode()"""

    embedding: np.ndarray
    input_token_count: int


class TruncatedTokensTuple(NamedTuple):
    """Output of SentenceTransformerWithTruncate._truncate_input_tokens()"""

    tokenized: BatchEncoding
    input_token_count: int
    truncation_needed: List[int]


# pylint: disable=too-many-lines
@module(
    "eeb12558-b4fa-4f34-a9fd-3f5890e9cd3f",
    "EmbeddingModule",
    "0.0.1",
    tasks=[
        EmbeddingTask,
        EmbeddingTasks,
        SentenceSimilarityTask,
        SentenceSimilarityTasks,
        RerankTask,
        RerankTasks,
        TokenizationTask,
    ],
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

        # Read config/env settings that are needed at run_* time.
        embedding_cfg = get_config().get("embedding", {})

        self.autocast = env_val_to_bool(embedding_cfg.get("autocast"))
        self.no_implicit_truncation = env_val_to_bool(
            embedding_cfg.get("implicit_truncation_errors", True)
        )

        self.batch_size = embedding_cfg.get("batch_size", 0)
        error.type_check("<NLP83816537E>", int, EMBEDDING_BATCH_SIZE=self.batch_size)

        # Retry count if enabled to try again (was for thread contention errors)
        retries = embedding_cfg.get("retries", 0)
        error.type_check("<NLP41910524E>", int, EMBEDDING_RETRIES=retries)
        self.retry_count = max(
            retries, 0
        )  # Ensure non-negative, before using in loop! (treat <0 as zero)

    @classmethod
    def load(
        cls, model_path: Union[str, ModuleConfig], *args, **kwargs
    ) -> "EmbeddingModule":
        """Load model

        Args:
            model_path (Union[str, ModuleConfig]): Path to saved model or
                in-memory ModuleConfig

        Returns:
            EmbeddingModule
                Instance of this class built from the model.
        """

        config = ModuleConfig.load(model_path)
        error.dir_check("<NLP19403057E>", config.model_path)

        artifacts_path = config.get(cls._ARTIFACTS_PATH_KEY)
        error.value_check(
            "<NLP07391618E>",
            artifacts_path,
            ValueError(f"Model config missing '{cls._ARTIFACTS_PATH_KEY}'"),
        )

        artifacts_path = os.path.abspath(
            os.path.join(config.model_path, artifacts_path)
        )
        error.dir_check("<NLP34197772E>", artifacts_path)

        # Read config/env settings that are needed at load time.
        embedding_cfg = get_config().get("embedding", {})

        autocast = env_val_to_bool(embedding_cfg.get("autocast"))
        pt2_compile = env_val_to_bool(embedding_cfg.get("pt2_compile"))
        trust_remote_code = env_val_to_bool(embedding_cfg.get("trust_remote_code"))
        ipex = cls._get_ipex(env_val_to_bool(embedding_cfg.get("ipex")))
        device = cls._select_device(ipex, embedding_cfg.get("device", ""))

        model = SentenceTransformerWithTruncate(
            model_name_or_path=artifacts_path,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        model.eval()  # required for IPEX at least
        if device is not None:
            model.to(torch.device(device))
        model = EmbeddingModule._optimize(model, ipex, device, autocast, pt2_compile)
        return cls(model)

    @property
    def public_model_info(cls) -> Dict[str, Any]:  # pylint: disable=no-self-argument
        """Helper property to return public metadata about a specific Model. This
        function is separate from `metdata` as that contains the entire ModelConfig
        which might not want to be shared/exposed.

        Returns:
            Dict[str, str]: A dictionary of this models's public metadata
        """
        return {
            "max_seq_length": cls.model.max_seq_length,
            "sentence_embedding_dimension": cls.model.get_sentence_embedding_dimension(),
        }

    @TokenizationTask.taskmethod()
    def run_tokenizer(
        self,
        text: str,
    ) -> TokenizationResults:
        """Run tokenization task against the model

        Args:
            text: str
                Text to tokenize
        Returns:
            TokenizationResults
                The token count
        """
        result = self.model._get_tokenized([text])

        mapping = [
            interv for interv in result.offset_mapping[0] if (interv[1] - interv[0]) > 0
        ]
        tokens = [Token(start=i[0], end=i[1], text=text[i[0] : i[1]]) for i in mapping]

        return TokenizationResults(token_count=len(result.input_ids[0]), results=tokens)

    @classmethod
    def _get_ipex(cls, ipex_flag):
        """Get IPEX optimization library if enabled and available, else return False

        Returns ipex library or False
        """
        ret = False

        # Enabled by environment variable
        # When IPEX is not false, attempt to import the library and use it.
        if ipex_flag:
            try:
                ret = importlib.import_module("intel_extension_for_pytorch")
            except Exception as ie:  # pylint: disable=broad-exception-caught
                # We don't require the module so catch, log, proceed to return False
                msg = (
                    f"IPEX enabled in env, but skipping ipex.optimize() because "
                    f"import intel_extension_for_pytorch failed with exception: {ie}"
                )
                logger.warning(msg, exc_info=True)

        return ret

    @staticmethod
    def _select_device(use_ipex, device):
        """Use environment variables and availability to determine the device to use"""
        if use_ipex:
            # If enabled, use "xpu" (IPEX on GPU instead of IPEX on CPU)
            if device == "xpu":
                return "xpu"
        elif device == "mps" and mps.is_built() and mps.is_available():
            # Never use on ipex, but otherwise use mps if enabled and available
            return "mps"

        return "cuda" if torch.cuda.is_available() else None

    @staticmethod
    def _get_backend(use_ipex, use_device):
        """Determine the backend to use for torch compile.

        Considers global ipex if enabled first, next mps device, finally defaults.

        Returns the backend for torch.compile()
        """
        if use_ipex:
            return "ipex"
        if use_device == "mps":
            return mps
        return "inductor"  # default backend

    @staticmethod
    def _optimize(model, ipex, device, autocast, pt2_compile):
        if ipex:
            if autocast:  # IPEX performs best with autocast using bfloat16
                model = ipex.optimize(
                    model, dtype=torch.bfloat16, weights_prepack=False
                )
            else:
                model = ipex.optimize(model, weights_prepack=False)

        # torch.compile won't work everywhere, but when set we'll try it
        if pt2_compile:
            backend = EmbeddingModule._get_backend(ipex, device)
            try:
                model = torch.compile(model, backend=backend, mode="max-autotune")
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Not always supported (e.g. in a python version) so catch, log, proceed.
                warn_msg = (
                    f"PT2_COMPILE enabled, but continuing without torch.compile() "
                    f"because it failed with exception: {e}"
                )
                logger.warning(warn_msg, exc_info=True)
        return model

    def _with_retry(self, fn: Callable[..., RT], *args, **kwargs) -> RT:
        first_exception = None
        for count in range(1 + self.retry_count):  # try once plus retries (if needed)
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # pylint: disable=broad-exception-caught
                if first_exception is None:
                    first_exception = e
                if self.retry_count > 0:
                    warn_msg = f"Try {count + 1}: {fn} failed due to: {e}"
                    logger.warning("<NLP54902271W>", warn_msg, exc_info=True)
                    if count + 1 < self.retry_count:
                        time.sleep(0.1 * (count * 2))

        # If above return did not happen, raise the first exception
        error.log_raise(
            log_code="<NLP13096081E>",
            exception=first_exception,
        )

    def _encode_with_retry(
        self, *args, **kwargs
    ) -> Union[EmbeddingResultTuple, List[torch.Tensor], np.ndarray, torch.Tensor]:
        """All encode calls should use this for consistent param adding and retry loop"""

        # Add the batch_size kwarg if not passed in and given a usable BATCH_SIZE
        if self.batch_size > 0:
            if kwargs is None:
                kwargs = {}
            if "batch_size" not in kwargs:
                kwargs["batch_size"] = self.batch_size

        if isinstance(self.model, SentenceTransformerWithTruncate):
            kwargs[
                "implicit_truncation_errors"
            ] = self.no_implicit_truncation  # config/env overrides default
            kwargs["autocast"] = self.autocast  # config/env overrides default
            return self._with_retry(self.model.encode, *args, **kwargs)

        # Else...
        # It's possible to init with a model that doesn't have the added kwargs.
        # E.g. a SentenceTransformer or other transformer model. Remove those kwargs!
        # This is not the normal use case but at least don't pass invalid kwargs, to encode()
        # and don't return the unexpected tuple (adding token count).
        if "truncate_input_tokens" in kwargs:
            del kwargs["truncate_input_tokens"]
        if "return_token_count" in kwargs:
            del kwargs["return_token_count"]
        if "implicit_truncation_errors" in kwargs:
            del kwargs["implicit_truncation_errors"]
        if "autocast" in kwargs:
            del kwargs["autocast"]
        return self._with_retry(self.model.encode, *args, **kwargs)

    @EmbeddingTask.taskmethod()
    def run_embedding(
        self,
        text: str,
        truncate_input_tokens: Optional[int] = 0,
    ) -> EmbeddingResult:
        """Get embedding for a string.
        Args:
            text: str
                Input text to be processed
            truncate_input_tokens: int
                Truncation length for input tokens.
                If less than zero, this is disabled (returns texts without processing).
                If zero or greater than the model's maximum, then this is a test
                to see if truncation is needed. If needed, an exception is thrown.
                Otherwise, we take this usable truncation limit to truncate the tokens and
                decode them to return truncated strings that can be used with this model.
        Returns:
            EmbeddingResult: the result vector nicely wrapped up
        """
        error.type_check("<NLP27491611E>", str, text=text)

        embeddings, input_token_count = self._encode_with_retry(
            text,
            truncate_input_tokens=truncate_input_tokens,
            return_token_count=True,
        )
        return EmbeddingResult(
            result=Vector1D.from_vector(embeddings),
            producer_id=self.PRODUCER_ID,
            input_token_count=input_token_count,
        )

    @EmbeddingTasks.taskmethod()
    def run_embeddings(
        self, texts: List[str], truncate_input_tokens: Optional[int] = 0, **kwargs
    ) -> EmbeddingResults:
        """Get embedding vectors for texts.
        Args:
            texts: List[str]
                List of input texts to be processed
            truncate_input_tokens: int
                Truncation length for input tokens.
                If less than zero, this is disabled (returns texts without processing).
                If zero or greater than the model's maximum, then this is a test
                to see if truncation is needed. If needed, an exception is thrown.
                Otherwise, we take this usable truncation limit to truncate the tokens and then
                decode them to return truncated strings that can be used with this model.
        Returns:
            EmbeddingResults: List of vectors. One for each input text (in order).
             Each vector is a list of floats (supports various float types).
        """
        if isinstance(
            texts, str
        ):  # encode allows str, but the result would lack a dimension
            texts = [texts]

        embeddings, input_token_count = self._encode_with_retry(
            texts,
            truncate_input_tokens=truncate_input_tokens,
            return_token_count=True,
            **kwargs,
        )
        vectors = [Vector1D.from_vector(e) for e in embeddings]

        return EmbeddingResults(
            results=ListOfVector1D(vectors=vectors),
            producer_id=self.PRODUCER_ID,
            input_token_count=input_token_count,
        )

    @SentenceSimilarityTask.taskmethod()
    def run_sentence_similarity(
        self,
        source_sentence: str,
        sentences: List[str],
        truncate_input_tokens: Optional[int] = 0,
        **kwargs,
    ) -> SentenceSimilarityResult:
        """Get similarity scores for each of sentences compared to the source_sentence.
        Args:
            source_sentence: str
            sentences: List[str]
                Sentences to compare to source_sentence
            truncate_input_tokens: int
                Truncation length for input tokens.
                If less than zero, this is disabled (returns texts without processing).
                If zero or greater than the model's maximum, then this is a test
                to see if truncation is needed. If needed, an exception is thrown.
                Otherwise, we take this usable truncation limit to truncate the tokens and then
                decode them to return truncated strings that can be used with this model.
        Returns:
            SentenceSimilarityResult: Similarity scores for each sentence.
        """

        source_embedding, source_token_count = self._encode_with_retry(
            source_sentence,
            truncate_input_tokens=truncate_input_tokens,
            return_token_count=True,
            **kwargs,
        )
        embeddings, sentences_token_count = self._encode_with_retry(
            sentences,
            truncate_input_tokens=truncate_input_tokens,
            return_token_count=True,
            **kwargs,
        )

        input_token_count = source_token_count + sentences_token_count
        res = cos_sim(source_embedding, embeddings)

        return SentenceSimilarityResult(
            result=SentenceSimilarityScores(scores=res.tolist()[0]),
            producer_id=self.PRODUCER_ID,
            input_token_count=input_token_count,
        )

    @SentenceSimilarityTasks.taskmethod()
    def run_sentence_similarities(
        self,
        source_sentences: List[str],
        sentences: List[str],
        truncate_input_tokens: Optional[int] = 0,
    ) -> SentenceSimilarityResults:
        """Run sentence-similarities on model.
        Args:
            source_sentences: List[str]
            sentences: List[str]
                Sentences to compare to source_sentences
            truncate_input_tokens: int
                Truncation length for input tokens.
                If less than zero, this is disabled (returns texts without processing).
                If zero or greater than the model's maximum, then this is a test
                to see if truncation is needed. If needed, an exception is thrown.
                Otherwise, we take this usable truncation limit to truncate the tokens and then
                decode them to return truncated strings that can be used with this model.
        Returns:
            SentenceSimilarityResults: Similarity scores for each source sentence in order.
                Each one contains the source-sentence's score for each sentence in order.
        """

        source_embedding, source_token_count = self._encode_with_retry(
            source_sentences,
            truncate_input_tokens=truncate_input_tokens,
            return_token_count=True,
        )
        embeddings, sentences_token_count = self._encode_with_retry(
            sentences,
            truncate_input_tokens=truncate_input_tokens,
            return_token_count=True,
        )

        input_token_count = source_token_count + sentences_token_count
        res = cos_sim(source_embedding, embeddings)
        float_list_list = res.tolist()

        return SentenceSimilarityResults(
            results=[SentenceSimilarityScores(fl) for fl in float_list_list],
            producer_id=self.PRODUCER_ID,
            input_token_count=input_token_count,
        )

    @RerankTask.taskmethod()
    def run_rerank_query(
        self,
        query: str,
        documents: List[JsonDict],
        top_n: Optional[int] = None,
        truncate_input_tokens: Optional[int] = 0,
        return_documents: bool = True,
        return_query: bool = True,
        return_text: bool = True,
        **kwargs,
    ) -> RerankResult:
        """Rerank the documents returning the most relevant top_n in order for this query.
        Args:
            query: str
                Query is the source string to be compared to the text of the documents.
            documents:  List[JsonDict]
                Each document is a dict. The text value is used for comparison to the query.
                If there is no text key, then _text is used and finally default is "".
            top_n:  Optional[int]
                Results for the top n most relevant documents will be returned.
                If top_n is not provided or (not > 0), then all are returned.
            truncate_input_tokens: int
                Truncation length for input tokens.
                If less than zero, this is disabled (returns texts without processing).
                If zero or greater than the model's maximum, then this is a test
                to see if truncation is needed. If needed, an exception is thrown.
                Otherwise, we take this usable truncation limit to truncate the tokens and then
                decode them to return truncated strings that can be used with this model.
            return_documents:  bool
                Default True
                Setting to False will disable returning of the input document (index is returned).
            return_query:  bool
                Default True
                Setting to False will disable returning of the query (results are in query order)
            return_text:  bool
                Default True
                Setting to False will disable returning of document text string that was used.
        Returns:
            RerankResult
                Returns the (top_n) scores in relevance order (most relevant first).
                The results always include a score and index which may be used to find the document
                in the original documents list. Optionally, the results also contain the entire
                document with its score (for use in chaining) and for convenience the query and
                text used for comparison may be returned.

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
            truncate_input_tokens=truncate_input_tokens,
            return_documents=return_documents,
            return_queries=return_query,
            return_text=return_text,
            **kwargs,
        )

        if results.results:
            return RerankResult(
                result=results.results[0],
                producer_id=self.PRODUCER_ID,
                input_token_count=results.input_token_count,
            )

        RerankResult(
            result=RerankScore(
                scores=[],
                query=query if return_query else None,
            ),
            producer_id=self.PRODUCER_ID,
            input_token_count=results.input_token_count,
        )

    @RerankTasks.taskmethod()
    def run_rerank_queries(
        self,
        queries: List[str],
        documents: List[JsonDict],
        top_n: Optional[int] = None,
        truncate_input_tokens: Optional[int] = 0,
        return_documents: bool = True,
        return_queries: bool = True,
        return_text: bool = True,
        **kwargs,
    ) -> RerankResults:
        """Rerank the documents returning the most relevant top_n in order for each of the queries.
        Args:
            queries: List[str]
                Each of the queries will be compared to the text of each of the documents.
            documents:  List[JsonDict]
                Each document is a dict. The text value is used for comparison to the query.
                If there is no text key, then _text is used and finally default is "".
            top_n:  Optional[int]
                Results for the top n most relevant documents will be returned.
                If top_n is not provided or (not > 0), then all are returned.
            truncate_input_tokens: int
                Truncation length for input tokens.
                If less than zero, this is disabled (returns texts without processing).
                If zero or greater than the model's maximum, then this is a test
                to see if truncation is needed. If needed, an exception is thrown.
                Otherwise, we take this usable truncation limit to truncate the tokens and then
                decode them to return truncated strings that can be used with this model.
            return_documents:  bool
                Default True
                Setting to False will disable returning of the input document (index is returned).
            return_queries:  bool
                Default True
                Setting to False will disable returning of the query (results are in query order)
            return_text:  bool
                Default True
                Setting to False will disable returning of document text string that was used.
        Returns:
            RerankResults
                For each query in queries (in the original order)...
                Returns the (top_n) scores in relevance order (most relevant first).
                The results always include a score and index which may be used to find the document
                in the original documents list. Optionally, the results also contain the entire
                document with its score (for use in chaining) and for convenience the query and
                text used for comparison may be returned.
        """

        error.type_check(
            "<NLP09038249E>",
            list,
            queries=queries,
            documents=documents,
        )

        error.value_check(
            "<NLP24788937E>",
            queries and documents,
            "Cannot rerank without a query and at least one document",
        )

        if top_n is None or top_n < 1:
            top_n = len(documents)

        # Using input document dicts so get "text" else "_text" else default to ""
        def get_text(doc):
            return doc.get("text") or doc.get("_text", "")

        doc_texts = [get_text(doc) for doc in documents]

        doc_embeddings, doc_token_count = self._encode_with_retry(
            doc_texts,
            truncate_input_tokens=truncate_input_tokens,
            return_token_count=True,
            convert_to_tensor=True,
            **kwargs,
        )
        doc_embeddings = normalize(doc_embeddings.to(self.model.device))

        query_embeddings, query_token_count = self._encode_with_retry(
            queries,
            truncate_input_tokens=truncate_input_tokens,
            return_token_count=True,
            convert_to_tensor=True,
            **kwargs,
        )
        query_embeddings = normalize(query_embeddings.to(self.model.device))

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
            RerankScores(
                query=add_query(q),
                scores=[RerankScore(**x) for x in r],
            )
            for q, r in enumerate(res)
        ]
        input_token_count = doc_token_count + query_token_count

        return RerankResults(
            results=results,
            producer_id=self.PRODUCER_ID,
            input_token_count=input_token_count,
        )

    @classmethod
    def bootstrap(cls, *args, **kwargs) -> "EmbeddingModule":
        """Bootstrap a sentence-transformers model

        Args:
            kwargs are passed to SentenceTransformer(**kwargs)
        """

        if "trust_remote_code" not in kwargs:
            # Read config/env settings that are needed at bootstrap time.
            embedding_cfg = get_config().get("embedding", {})
            kwargs["trust_remote_code"] = env_val_to_bool(
                embedding_cfg.get("trust_remote_code")
            )

        return cls(model=SentenceTransformer(*args, **kwargs))

    def save(self, model_path: str, *args, **kwargs):
        """Save model using config in model_path

        Args:
            model_path: str
                Path to model config
        """

        error.type_check("<NLP82314992E>", str, model_path=model_path)
        model_config_path = model_path.strip()
        error.value_check(
            "<NLP40145207E>",
            model_config_path,
            f"model_path '{model_config_path}' is invalid",
        )

        model_config_path = os.path.abspath(
            model_config_path.strip()
        )  # No leading/trailing spaces sneaky weirdness

        # Only allow new dirs because there are not enough controls to safely update in-place
        os.makedirs(model_config_path, exist_ok=False)

        saver = ModuleSaver(
            module=self,
            model_path=model_config_path,
        )
        artifacts_path = self._ARTIFACTS_PATH_DEFAULT
        saver.update_config({self._ARTIFACTS_PATH_KEY: artifacts_path})

        # Save the model
        self.model.save(os.path.join(model_config_path, artifacts_path))

        # Save the config
        ModuleConfig(saver.config).save(model_config_path)


def get_sample_start_indexes(tokenized: BatchEncoding) -> List[int]:
    """Returns a list containing the index for the first encoding of each sample
    contained in tokenized."""

    # When truncating occurs a sample is split across multiple encodings
    # ie. len(tokenized.encodings) > the number of text samples input for tokenization

    # Knowing the encoding index of where each sample's first encoding is located allows us to
    # access the encodings for individual samples

    # note: tokenized["overflow_to_sample_mapping"] is a torch.Tensor

    samples_start_indexes: Dict[int, int] = {}
    for i, tensor_sample in enumerate(tokenized["overflow_to_sample_mapping"]):
        int_sample = int(tensor_sample)
        if int_sample not in samples_start_indexes:
            samples_start_indexes[int_sample] = i

    return list(samples_start_indexes.values())


class TruncateCountBehavior(Enum):
    ONLY = auto()
    ALL = auto()
    IGNORE = auto()


def sum_token_count(
    tokenized: BatchEncoding,
) -> int:
    """Returns the number of non-special tokens. Assumes truncation w/o overflow.
    Args:
        tokenized: BatchEncoding
    Returns:
        Int total of all tokens contained in tokenized.
    """
    # Encoding objects have various attributes of note:
    # - tokens: list of tokens (sub-parts of the input strings after word/subword
    #       splitting and before conversion to integer indices)
    # - attention_mask: List of indices specifying which tokens should be attended to
    #       by the model. Note that [PAD] = 0, while [CLS] / [SEP] = 1
    # - special_tokens_mask: List of 0s and 1s, with 1 specifying added special tokens
    #       and 0 specifying regular sequence tokens

    error.type_check(
        "<NLP82314993E>",
        BatchEncoding,
        tokenized=tokenized,
    )
    error.value_check(
        "<NLP82314995E>",
        tokenized.encodings,
        "Number of tokenized encodings is only known when a non-python tokenizer is used",
    )

    token_count = 0

    # Sum the length of all encodings for all samples
    for encoding in tokenized.encodings:
        token_count += sum(encoding.attention_mask)

    return token_count


def _truncate_texts(texts, tokenized, max_length, text_indexes):
    """Truncate texts using tokenized offsets and desired max_length.

    This implements truncation in the texts without changing the tokenizer
    truncation parameters to avoid thread locking ("Already borrowed" exceptions).

    After the texts have been truncated, they should be re-tokenized
    to get a new `tokenized` structure for use in encode.
    """

    for text_number in text_indexes:

        # The offset_mapping describes the text position for each token in this text.
        offsets = tokenized["offset_mapping"][text_number]

        # Find the first offset that is not empty (0, 0) to avoid added tokens
        # Note: Normally just start at 0, but it's imaginable that tokenizer could skip stuff.
        start = next(offset for offset in offsets if offset != (0, 0))[0]
        # Find the end index where the text string should be truncated
        end = _get_end_index(max_length, text_number, tokenized)

        # Use the start-beginning end-ending to slice the text based on token truncation
        # i.e. if start=(0,5) and end=(72,78) then we want slice [0:78]
        texts[text_number] = texts[text_number][
            start:end
        ]  # replace text with truncated


def _get_end_index(max_length, text_number, tokenized):
    offsets = tokenized["offset_mapping"][text_number]
    attn_mask = tokenized.encodings[text_number].attention_mask

    # Find the last offset by counting attn masks
    # and keeping the last non-zero offset end.
    token_count = 0
    end_index = 0
    for n, attn in enumerate(attn_mask):
        if attn == 1:
            token_count += 1
            end = offsets[n][1]  # Index to end character from offset
            if end > end_index:  # Grab last non-zero end index (ensures increasing too)
                end_index = end
        if token_count >= max_length - 1:  # Stop with room for an end token
            break
    return end_index


class SentenceTransformerWithTruncate(SentenceTransformer):
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        modules: Optional[Iterable[nn.Module]] = None,
        device: Optional[str] = None,
        prompts: Optional[Dict[str, str]] = None,
        default_prompt_name: Optional[str] = None,
        similarity_fn_name: Optional[Union[str, SimilarityFunction]] = None,
        cache_folder: Optional[str] = None,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        token: Optional[Union[bool, str]] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        truncate_dim: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        model_card_data: Optional[SentenceTransformerModelCardData] = None,
    ):
        super().__init__(
            model_name_or_path,
            modules,
            device,
            prompts,
            default_prompt_name,
            similarity_fn_name,
            cache_folder,
            trust_remote_code,
            revision,
            local_files_only,
            token,
            use_auth_token,
            truncate_dim,
            model_kwargs,
            tokenizer_kwargs,
            config_kwargs,
            model_card_data,
        )
        self.tokenizers = {}

    def _truncation_needed(self, tokenized, max_length, texts):
        """Check for truncation needed to meet max_length token limit
        Returns:
            List of indexes of the texts that need truncating ([] if none)
        """

        ret = []  # List of indexes for texts that need truncation

        if max_length < 0:
            # -1 means to just let the model do its thing
            return ret

        for i, encoding in enumerate(tokenized.encodings):
            input_tokens = sum(encoding.attention_mask)
            if input_tokens > max_length or input_tokens > self.max_seq_length:
                # Greater than truncate_input_tokens plus 2 (start/end) or over model limit
                ret.append(i)
            elif input_tokens == self.max_seq_length:
                # At model limit, including start/end...
                # This may or may not have already been truncated at the model limit.
                # Check the strlen and last offset to see if the text actually needs truncating
                # to make room for the end separator token.
                # We need to know this, for "not okay_to_truncate" errors.
                end_index = _get_end_index(max_length, i, tokenized)
                if end_index < len(texts[i]):
                    ret.append(i)

        return ret

    def _tokenize_plus(
        self,
        truncate_input_tokens: int,
        texts: List[str],
        implicit_truncation_errors: bool = True,
        **kwargs,
    ) -> TruncatedTokensTuple:
        """Tokenize with support for truncation handling and returning the token count
        Args:
            truncate_input_tokens: int
                Truncation length for input tokens.
                If less than zero, this truncation is left up to the tokenizer default (model max).
                If zero or greater than the model's maximum, then this is used as a test
                to see if truncation is needed. If needed is needed, an exception is thrown.
                Otherwise, we take this usable truncation limit to truncate the input tokens.
            texts: List[str]
                Input texts to be checked and optionally truncated.
            implicit_truncation_errors: bool
                Configuration indicates whether implicit truncation should be rejected.
        Returns:
            Tuple containing a dictionary of lists/arrays/tensors returned by the tokenizer, with
            proper truncation ('input_ids', 'attention_mask', etc.), and the input_token_count int.
        """

        max_tokens = self.max_seq_length

        # Do truncation if given a usable truncation value, else test for need to truncation
        if truncate_input_tokens < 0:
            okay_to_truncate = True
            max_length = max_tokens
        elif 0 < truncate_input_tokens <= max_tokens:
            okay_to_truncate = True
            # Add 2 for begin/end tokens, but don't go higher than model's max_tokens
            max_length = min(truncate_input_tokens + 2, max_tokens)

        else:
            okay_to_truncate = not implicit_truncation_errors
            max_length = max_tokens

        assert len(texts) > 0, "Cannot truncate nothing"
        assert isinstance(texts[0], str), "Only str can be truncated"

        texts = [str(s).strip() for s in texts]

        # Call tokenizer with the same truncation parameters every time
        tokenized = self._get_tokenized(texts, **kwargs)

        # Custom truncation and/or error raise if needed
        truncation_needed = self._truncation_needed(tokenized, max_length, texts)
        if truncation_needed and okay_to_truncate:
            # Truncate texts in place
            _truncate_texts(texts, tokenized, max_length, truncation_needed)
            # Re-tokenize the truncated texts
            tokenized = self._get_tokenized(texts, **kwargs)
            truncation_needed = []  # truncation accomplished

        input_token_count = sum_token_count(tokenized)
        return TruncatedTokensTuple(tokenized, input_token_count, truncation_needed)

    def _get_tokenized(self, texts, **kwargs):
        """Intentionally always call tokenizer the same way to avoid thread issues.

        Use a copy of the tokenizer per-model (self) and per-thread (map by thread ID).

        Avoid changing the max length, truncation, and padding to avoid the
        "Already borrowed" errors that come with concurrent threads attempting to use
        the fast tokenizer with different truncation settings.
        """

        padding_strategy = kwargs.pop("padding_strategy", True)

        # Keep copies of tokenizer per thread (in each wrapped model instance)
        thread_id = threading.get_ident()
        tokenizer = (
            self.tokenizers[thread_id]
            if thread_id in self.tokenizers
            else self.tokenizers.setdefault(thread_id, deepcopy(self.tokenizer))
        )

        return tokenizer(
            texts,
            return_attention_mask=True,  # Used for determining token count
            return_token_type_ids=False,
            return_overflowing_tokens=False,  # DO NOT USE overflow tokens break sentence batches
            return_offsets_mapping=True,  # Used for truncation
            return_length=False,
            return_tensors="pt",
            truncation=True,  # DO NOT CHANGE else "Already borrowed" errors
            padding=padding_strategy,  # DO NOT CHANGE else "Already borrowed" errors
            max_length=self.max_seq_length,  # DO NOT CHANGE else "Already borrowed" errors
        )

    def encode(
        self,
        sentences: Union[str, List[str]],
        prompt_name: Optional[str] = None,
        prompt: Optional[str] = None,
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
        truncate_input_tokens: int = 0,
        return_token_count: bool = False,
        implicit_truncation_errors: bool = True,
        autocast: bool = False,
        **kwargs,
    ) -> Union[EmbeddingResultTuple, List[torch.Tensor], np.ndarray, torch.Tensor]:
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param prompt_name: Ignored here. Added for compatibility with super API.
        :param prompt: Ignored here. Added for compatibility with super API.
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Ignored here. Added for compatibility with super API.
        :param output_value: Ignored here. Added for compatibility with super API.
        :param precision: Ignored here. Added for compatibility with super API.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list
                of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any
                setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: Ignored here. Added for compatibility with super API.
        :param truncate_input_tokens: Truncation length for input tokens.
                Truncation length for input tokens.
                If less than zero, this truncation is left up to the tokenizer default (model max).
                If zero or greater than the model's maximum, then this is used as a test
                to see if truncation is needed. If truncation is needed, an exception is thrown,
                unless implicit_truncation_errors=False (see below).
                Otherwise, we take this usable truncation limit to truncate the input tokens.
        :param return_token_count: If true, a tuple is returned to add the input token count.
        :param implicit_truncation_errors: If true (default) implicit truncation throws an error.
                If false, the model default behavior or used.
        :param autocast: If true (not default) run with torch.cpu.amp.autocast()

        :return:
           If return_token_count is False, the embedding is returned as a numpy matrix.
           If return_token_count is True, a tuple is returned with both the embedding and
                the input token count.
        """

        # These args are for API compatability, but are currently ignored in our version of encode()
        _ = (
            prompt_name,
            prompt,
            show_progress_bar,
            output_value,
            precision,
            normalize_embeddings,
        )

        self.eval()

        if convert_to_tensor:
            convert_to_numpy = False

        input_was_string = False
        list_of_sentences = sentences
        if isinstance(list_of_sentences, str) or not isinstance(
            sentences, Sized
        ):  # Cast an individual sentence to a list with length 1
            list_of_sentences = [sentences]
            input_was_string = True

        error.type_check_all("<NLP82314994E>", str, sentences=list_of_sentences)

        if device is None:
            device = self.device

        self.to(device)

        all_embeddings = []

        # Sort sentences according to length, from longest to shortest
        # OOM errors then occurs at start of encoding
        length_sorted_idx = np.argsort(
            [-self._text_length(sen) for sen in list_of_sentences]
        )
        sentences_sorted: list[str] = [
            list_of_sentences[idx] for idx in length_sorted_idx
        ]

        input_token_count = 0

        for start_index in range(0, len(list_of_sentences), batch_size):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features, token_count, truncation_needed = self._tokenize_plus(
                truncate_input_tokens,
                sentences_batch,
                implicit_truncation_errors=implicit_truncation_errors,
                **kwargs,
            )

            if truncation_needed:  # truncation was needed and was not done/not allowed
                if input_was_string:
                    index_hint = "."
                else:
                    # Add index hint for texts where the error was detected.
                    # Adjust indexes for the start of the batch
                    truncation_needed = [x + start_index for x in truncation_needed]
                    # Convert index to pre-sorted index
                    truncation_needed = [
                        length_sorted_idx[x] for x in truncation_needed
                    ]
                    indexes = f"{', '.join(str(i) for i in truncation_needed)}."
                    index_hint = (
                        " for text at "
                        f"{'index' if len(truncation_needed) == 1 else 'indexes'}: {indexes}"
                    )

                error.log_raise(
                    "<NLP08391926E>",
                    ValueError(
                        f"Token sequence length (+2 for start/end tokens) exceeds the "
                        f"maximum sequence length for this model ({self.max_seq_length})"
                        f"{index_hint}"
                    ),
                )

            input_token_count += token_count

            features = batch_to_device(features, device)

            if autocast:
                with torch.no_grad(), torch.cpu.amp.autocast():
                    out_features = self.forward(features)
                    embeddings = out_features["sentence_embedding"]
                    if convert_to_numpy:
                        embeddings = embeddings.detach().cpu()
                    all_embeddings.extend(embeddings)
            else:
                with torch.no_grad():
                    out_features = self.forward(features)
                    embeddings = out_features["sentence_embedding"]
                    if convert_to_numpy:
                        embeddings = embeddings.detach().cpu()
                    all_embeddings.extend(embeddings)

        # Restore original order
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return (
            EmbeddingResultTuple(all_embeddings, input_token_count)
            if return_token_count
            else all_embeddings
        )
