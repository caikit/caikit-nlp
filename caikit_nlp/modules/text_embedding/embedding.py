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
from copy import deepcopy
from typing import List, Optional
import importlib
import os
import time

# Third Party
from torch.backends import mps
import torch

# First Party
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
)
from caikit.interfaces.nlp.tasks import (
    EmbeddingTask,
    EmbeddingTasks,
    RerankTask,
    RerankTasks,
    SentenceSimilarityTask,
    SentenceSimilarityTasks,
)
import alog

logger = alog.use_channel("TXT_EMB")
error = error_handler.get(logger)

# To avoid dependency problems, make sentence-transformers an optional import and
# defer any ModuleNotFoundError until someone actually tries to init a model with this module.
try:
    sentence_transformers = importlib.import_module("sentence_transformers")
    # Third Party
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import (
        cos_sim,
        dot_score,
        normalize_embeddings,
        semantic_search,
    )
except ModuleNotFoundError:
    # When it is not available, create a dummy that raises an error on attempted init()
    class SentenceTransformerNotAvailable:
        def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
            # Will reproduce the ModuleNotFoundError if/when anyone actually tries this module/model
            importlib.import_module("sentence_transformers")

    SentenceTransformer = SentenceTransformerNotAvailable


# For testing env vars for values that mean false
FALSY = ("no", "n", "false", "0", "f", "off")


def env_var_to_int(name, default):
    """Returns the integer value of name env var or default value if None or invalid integer"""
    s = os.getenv(name, default)
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


# Batch size for encode() if <= 0 or invalid, the sentence-transformers default is used
BATCH_SIZE = env_var_to_int("BATCH_SIZE", default=0)


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
    ],
)
class EmbeddingModule(ModuleBase):

    # Retry count if enabled to try again (was for thread contention errors)
    RETRY_COUNT = max(env_var_to_int("RETRY_COUNT", default=0), 0)

    _ARTIFACTS_PATH_KEY = "artifacts_path"
    _ARTIFACTS_PATH_DEFAULT = "artifacts"

    def __init__(
        self,
        model: SentenceTransformer,
    ):
        super().__init__()
        self.model = model

        # Separate copy of tokenizer just for _truncate_input_tokens()
        # This avoids RuntimeError('Already borrowed') which is way too frequent
        # otherwise when using Python threads and tokenize for truncation followed
        # by sentence-transformers tokenize/encode.
        self._tokenizer = deepcopy(self.model.tokenizer)

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

        ipex = cls._get_ipex()
        device = cls._select_device(ipex)
        model = SentenceTransformer(model_name_or_path=artifacts_path, device=device)
        if device is not None:
            model.to(torch.device(device))
        model = EmbeddingModule._optimize(model, ipex, device)

        # Validate model with any encode test (simple and hardcoded for now).
        # This gets some of the first-time inference cost out of the way.
        # This avoids using the tokenizer (for truncation) before it is ready.
        model.encode("warmup")

        return cls(model)

    @staticmethod
    def _get_ipex():
        """Get IPEX optimization library if enabled and available, else return False

        Returns ipex library or False
        """
        ret = False

        # Enabled by environment variable
        # When IPEX_OPTIMIZE is not false, attempt to import the library and use it.
        if os.getenv("IPEX_OPTIMIZE", "false").lower() not in FALSY:
            try:
                ret = importlib.import_module("intel_extension_for_pytorch")
            except Exception as ie:  # pylint: disable=broad-exception-caught
                # We don't require the module so catch, log, proceed to return False
                msg = (
                    f"IPEX_OPTIMIZE enabled in env, but skipping ipex.optimize() because "
                    f"import intel_extension_for_pytorch failed with exception: {ie}"
                )
                logger.warning(msg, exc_info=1)

        return ret

    @staticmethod
    def _select_device(use_ipex):
        """Use environment variables and availability to determine the device to use"""
        if use_ipex:
            # If enabled, use "xpu" (IPEX on GPU instead of IPEX on CPU)
            if os.getenv("USE_XPU", "false").lower() not in FALSY:
                return "xpu"
        elif (
            os.getenv("USE_MPS", "false").lower() not in FALSY
            and mps.is_built()
            and mps.is_available()
        ):
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
    def _optimize(model, ipex, device):

        if ipex:
            model = ipex.optimize(model)

        # torch.compile won't work everywhere, but when set we'll try it
        if os.getenv("PT2_COMPILE", "false").lower() not in FALSY:
            backend = EmbeddingModule._get_backend(ipex, device)
            try:
                model = torch.compile(model, backend=backend, mode="max-autotune")
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Not always supported (e.g. in a python version) so catch, log, proceed.
                warn_msg = (
                    f"PT2_COMPILE enabled in env, but continuing without torch.compile() "
                    f"because it failed with exception: {e}"
                )
                logger.warning(warn_msg, exc_info=True)
        return model

    def _with_retry(self, fn, *args, **kwargs):
        first_exception = None
        for count in range(1 + self.RETRY_COUNT):  # try once plus retries (if needed)
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # pylint: disable=broad-exception-caught
                if first_exception is None:
                    first_exception = e
                if self.RETRY_COUNT > 0:
                    warn_msg = f"Try {count + 1}: {fn} failed due to: {e}"
                    logger.warning("<NLP54902271W>", warn_msg, exc_info=True)
                    if count + 1 < self.RETRY_COUNT:
                        time.sleep(0.1 * (count * 2))

        # If above return did not happen, raise the first exception
        error.log_raise(
            log_code="<NLP13096081E>",
            exception=first_exception,
        )

    def _encode_with_retry(self, *args, **kwargs):
        """All encode calls should use this for consistent param adding and retry loop"""

        # Add the batch_size kwarg if not passed in and given a usable BATCH_SIZE
        if BATCH_SIZE > 0:
            if kwargs is None:
                kwargs = {}
            if "batch_size" not in kwargs:
                kwargs["batch_size"] = BATCH_SIZE

        return self._with_retry(self.model.encode, *args, **kwargs)

    def _truncate_input_tokens(
        self, truncate_input_tokens, texts: List[str]
    ) -> List[str]:
        """Truncate input tokens
        Args:
            truncate_input_tokens: int
                Truncation length for input tokens.
                If less than zero, this is disabled (returns texts without processing).
                If zero or greater than the model's maximum, then this is a test
                to see if truncation is needed. If needed, an exception is thrown.
                Otherwise, we take this usable truncation limit to truncate the tokens and then
                decode them to return truncated strings that can be used with this model.
            texts: List[str]
                Input texts to be checked and optionally truncated.
        Returns:
            List[str]: the texts after checking and/or truncating
        """

        # NOTE: When inference is called immediately after load (typical case with lazy loading),
        # using the tokenizer right away here results in a warning like:
        #     huggingface/tokenizers: The current process just got forked, after parallelism
        #     has already been used. Disabling parallelism to avoid deadlocks...
        #     To disable this warning, you can either:
        #     - Avoid using `tokenizers` before the fork if possible
        #     - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
        # A warmup encode() call in load() will take care of this (the first option above).
        # This here comment is in case we need to set TOKENIZERS_PARALLELISM in the future.

        if truncate_input_tokens < 0:
            return texts

        max_tokens = self.model.max_seq_length

        # Do truncation if given a usable truncation value, else test for need to truncation
        if 0 < truncate_input_tokens <= max_tokens:
            okay_to_truncate = True
            max_length = truncate_input_tokens
            ret = []  # will return truncated texts
        else:
            okay_to_truncate = False
            max_length = max_tokens
            ret = texts  # will not alter texts when truncation is not allowed

        tokenized = self._with_retry(
            self._tokenizer,
            texts,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_length=True,
            truncation=True,
            max_length=max_length,
        )

        texts_map = tokenized["overflow_to_sample_mapping"]

        for text_number, text in enumerate(texts):
            # positions: the positions (in lengths and offsets arrays) that belong to this text
            positions = [
                position
                for position, sample_number in enumerate(texts_map)
                if sample_number == text_number
            ]
            lengths = [tokenized["length"][pos] for pos in positions]

            was_truncated = len(lengths) > 1  # multiple lengths when truncated

            if not okay_to_truncate and was_truncated:
                # Raise error. We don't allow silent truncation in this case.
                tokens = sum(lengths)  # add up total tokens for error message
                error.log_raise(
                    "<NLP08391926E>",
                    ValueError(
                        f"Token sequence length is longer than the specified "
                        f"maximum sequence length for this model ({tokens} > {max_tokens})."
                    ),
                )

            elif okay_to_truncate and not was_truncated:
                ret.append(text)  # collect original text to return

            elif okay_to_truncate and was_truncated:
                # Truncate the text that maps to the truncated tokens.
                # The offset_mapping describes the text position for each token.
                # Added tokens were not in the text, so they show up as (0, 0).

                # Get the text offsets for the tokens that are to be kept after truncation.
                # Take the first set of offsets that mapped to this text's positions.
                # This first set represents what will be kept after truncation.
                # Each offset tells us which chars in the original text map to this token.
                offsets = next(tokenized["offset_mapping"][pos] for pos in positions)

                # Find the first offset that is not empty (0, 0) to avoid added tokens
                start = next(offset for offset in offsets if offset != (0, 0))

                # Find the last offset that is not empty (0, 0) to avoid added tokens
                end = next(
                    offset for offset in reversed(list(offsets)) if offset != (0, 0)
                )

                # Use the start-beginning end-ending to slice the text based on token truncation
                # i.e. if start=(0,5) and end=(72,78) then we want slice [0:78]
                truncated_text = text[start[0] : end[1]]

                ret.append(truncated_text)  # return the truncated text for this one

        return ret

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

        text = self._truncate_input_tokens(truncate_input_tokens, [text])[0]
        return EmbeddingResult(
            result=Vector1D.from_vector(self._encode_with_retry(text)),
            producer_id=self.PRODUCER_ID,
        )

    @EmbeddingTasks.taskmethod()
    def run_embeddings(
        self,
        texts: List[str],
        truncate_input_tokens: Optional[int] = 0,
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

        texts = self._truncate_input_tokens(truncate_input_tokens, texts)

        embeddings = self._encode_with_retry(texts)
        vectors = [Vector1D.from_vector(e) for e in embeddings]
        return EmbeddingResults(
            results=ListOfVector1D(vectors=vectors), producer_id=self.PRODUCER_ID
        )

    @SentenceSimilarityTask.taskmethod()
    def run_sentence_similarity(
        self,
        source_sentence: str,
        sentences: List[str],
        truncate_input_tokens: Optional[int] = 0,
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

        source_sentence = self._truncate_input_tokens(
            truncate_input_tokens, [source_sentence]
        )[0]
        sentences = self._truncate_input_tokens(truncate_input_tokens, sentences)

        source_embedding = self._encode_with_retry(source_sentence)
        embeddings = self._encode_with_retry(sentences)

        res = cos_sim(source_embedding, embeddings)
        return SentenceSimilarityResult(
            result=SentenceSimilarityScores(scores=res.tolist()[0]),
            producer_id=self.PRODUCER_ID,
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

        source_sentences = self._truncate_input_tokens(
            truncate_input_tokens, source_sentences
        )
        sentences = self._truncate_input_tokens(truncate_input_tokens, sentences)

        source_embedding = self._encode_with_retry(source_sentences)
        embeddings = self._encode_with_retry(sentences)

        res = cos_sim(source_embedding, embeddings)
        float_list_list = res.tolist()
        return SentenceSimilarityResults(
            results=[SentenceSimilarityScores(fl) for fl in float_list_list],
            producer_id=self.PRODUCER_ID,
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
        ).results

        if results:
            return RerankResult(result=results[0], producer_id=self.PRODUCER_ID)

        RerankResult(
            producer_id=self.PRODUCER_ID,
            result=RerankScore(
                scores=[],
                query=query if return_query else None,
            ),
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

        doc_texts = self._truncate_input_tokens(truncate_input_tokens, doc_texts)
        queries = self._truncate_input_tokens(truncate_input_tokens, queries)

        doc_embeddings = normalize_embeddings(
            self._encode_with_retry(doc_texts, convert_to_tensor=True).to(
                self.model.device
            )
        )

        query_embeddings = normalize_embeddings(
            self._encode_with_retry(queries, convert_to_tensor=True).to(
                self.model.device
            )
        )

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

        return RerankResults(results=results, producer_id=self.PRODUCER_ID)

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
