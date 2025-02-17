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
from functools import partial
from typing import Any, Dict, List, NamedTuple, Optional, Union
import os
import threading

# Third Party
from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader
import numpy as np
import torch

# First Party
from caikit import get_config
from caikit.core import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.core.data_model.json_dict import JsonDict
from caikit.core.exceptions import error_handler
from caikit.interfaces.nlp.data_model import (
    RerankResult,
    RerankResults,
    RerankScore,
    RerankScores,
    Token,
    TokenizationResults,
)
from caikit.interfaces.nlp.tasks import RerankTask, RerankTasks, TokenizationTask
import alog

# Local
from caikit_nlp.modules.text_embedding.utils import env_val_to_bool

logger = alog.use_channel("CROSS_ENCODER")
error = error_handler.get(logger)


class RerankResultTuple(NamedTuple):
    """Output of modified rank()"""

    scores: list
    input_token_count: int


class PredictResultTuple(NamedTuple):
    """Output of modified predict()"""

    scores: np.ndarray
    input_token_count: int


# pylint: disable=too-many-lines disable=duplicate-code
@module(
    "1673f8f2-726f-48cb-93a1-540c81f0f3c9",
    "CrossEncoderModule",
    "0.0.1",
    tasks=[
        RerankTask,
        RerankTasks,
        TokenizationTask,
    ],
)
class CrossEncoderModule(ModuleBase):

    _ARTIFACTS_PATH_KEY = "artifacts_path"
    _ARTIFACTS_PATH_DEFAULT = "artifacts"

    def __init__(
        self,
        model: "CrossEncoderWithTruncate",
    ):
        super().__init__()
        self.model = model

        # model_max_length attribute availability might(?) vary by model/tokenizer
        self.model_max_length = getattr(model.tokenizer, "model_max_length", None)

        # Read config/env settings that are needed at run_* time.
        embedding_cfg = get_config().get("embedding", {})

        self.batch_size = embedding_cfg.get("batch_size", 32)
        error.type_check("<NLP83501588E>", int, EMBEDDING_BATCH_SIZE=self.batch_size)
        if self.batch_size <= 0:
            self.batch_size = 32  # 0 or negative, use the default.

    @classmethod
    def load(
        cls, model_path: Union[str, ModuleConfig], *args, **kwargs
    ) -> "CrossEncoderModule":
        """Load model

        Args:
            model_path (Union[str, ModuleConfig]): Path to saved model or
                in-memory ModuleConfig

        Returns:
            CrossEncoderModule
                Instance of this class built from the model.
        """

        config = ModuleConfig.load(model_path)
        error.dir_check("<NLP13823362E>", config.model_path)

        artifacts_path = config.get(cls._ARTIFACTS_PATH_KEY)
        error.value_check(
            "<NLP20896115E>",
            artifacts_path,
            f"Model config missing '{cls._ARTIFACTS_PATH_KEY}'",
        )

        artifacts_path = os.path.abspath(
            os.path.join(config.model_path, artifacts_path)
        )
        error.dir_check("<NLP33193321E>", artifacts_path)

        # Read config/env settings that are needed at load time.
        embedding_cfg = get_config().get("embedding", {})

        trust_remote_code = env_val_to_bool(embedding_cfg.get("trust_remote_code"))

        model = CrossEncoderWithTruncate(
            model_name=artifacts_path,
            trust_remote_code=trust_remote_code,
        )
        model.model.eval()
        model.model.to(model._target_device)

        return cls(model)

    @property
    def public_model_info(cls) -> Dict[str, Any]:  # pylint: disable=no-self-argument
        """Helper property to return public metadata about a specific Model. This
        function is separate from `metadata` as that contains the entire ModelConfig
        which might not want to be shared/exposed.

        Returns:
            Dict[str, str]: A dictionary of this model's public metadata
        """

        return (
            {"max_seq_length": cls.model_max_length}
            if cls.model_max_length is not None
            else {}
        )

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
        result = self.model.get_tokenized([text], return_offsets_mapping=True)

        mapping = [
            interv for interv in result.offset_mapping[0] if (interv[1] - interv[0]) > 0
        ]
        tokens = [Token(start=i[0], end=i[1], text=text[i[0] : i[1]]) for i in mapping]

        return TokenizationResults(token_count=len(result.input_ids[0]), results=tokens)

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
            "<NLP61983803E>",
            int,
            allow_none=True,
            top_n=top_n,
        )

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
        )

        return RerankResult(
            result=results.results[0],
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

        input_token_count = 0
        results = []
        for query in queries:
            scores, token_count = self.model.rank(
                query=query,
                documents=doc_texts,
                top_k=top_n,
                return_documents=False,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                truncate_input_tokens=truncate_input_tokens,
            )
            results.append(scores)
            input_token_count += token_count

        # Fixup result dicts
        for r in results:
            for x in r:
                x["score"] = float(x["score"].item())
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
            for q, r in enumerate(results)
        ]

        return RerankResults(
            results=results,
            producer_id=self.PRODUCER_ID,
            input_token_count=input_token_count,
        )

    @classmethod
    def bootstrap(cls, *args, **kwargs) -> "CrossEncoderModule":
        """Bootstrap a cross-encoder model

        Args:
            args/kwargs are passed to CrossEncoder
        """

        # Add ability to bootstrap with trust_remote_code using env var.
        if "trust_remote_code" not in kwargs:
            # Read config/env settings that are needed at bootstrap time.
            embedding_cfg = get_config().get("embedding", {})
            kwargs["trust_remote_code"] = env_val_to_bool(
                embedding_cfg.get("trust_remote_code")
            )

        return cls(model=CrossEncoder(*args, **kwargs))

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


class CrossEncoderWithTruncate(CrossEncoder):
    def __init__(
        self,
        model_name: str,
        num_labels: int = None,
        max_length: int = None,
        device: str = None,
        automodel_args: Dict = None,
        tokenizer_args: Dict = None,
        config_args: Dict = None,
        cache_dir: str = None,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        default_activation_function=None,
        classifier_dropout: float = None,
    ):
        super().__init__(
            model_name,
            num_labels,
            max_length,
            device,
            automodel_args,
            tokenizer_args,
            config_args,
            cache_dir,
            trust_remote_code,
            revision,
            local_files_only,
            default_activation_function,
            classifier_dropout,
        )
        self.tokenizers = {}

    def _get_tokenizer_per_thread(self):
        """Use a copy of the tokenizer per-model (self) and per-thread (map by thread ID)."""

        # Keep copies of tokenizer per thread (in each wrapped model instance)
        thread_id = threading.get_ident()
        tokenizer = (
            self.tokenizers[thread_id]
            if thread_id in self.tokenizers
            else self.tokenizers.setdefault(thread_id, deepcopy(self.tokenizer))
        )

        return tokenizer

    def get_tokenized(self, texts, **kwargs):
        """Use a copy of the tokenizer per-model (self) and per-thread (map by thread ID)"""

        max_len = kwargs.get("truncate_input_tokens", self.tokenizer.model_max_length)
        max_len = min(max_len, self.tokenizer.model_max_length)
        if max_len <= 0:
            max_len = None  # Use the default
        elif max_len < 5:
            # 1, 2, 3 don't really work (4 might but...)
            # Bare minimum is [CLS] token [SEP] token [SEP]
            max_len = 5

        tokenizer = self._get_tokenizer_per_thread()
        tokenized = tokenizer(
            *texts,
            return_attention_mask=True,  # Used for determining token count
            return_token_type_ids=False,  # Needed for cross-encoders
            return_overflowing_tokens=False,  # DO NOT USE overflow tokens break sentence batches
            return_offsets_mapping=True,  # Used for truncation needed error
            return_length=False,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_len,
        )
        return tokenized

    def _truncation_needed(self, encoding, texts):
        """Check for truncation needed to meet max_length token limit
        Returns:
            True if was truncated, False otherwise
        """

        input_tokens = sum(encoding.attention_mask)
        if input_tokens < self.tokenizer.model_max_length:
            return False

        # At model limit, including start/end...
        # This may or may not have already been truncated at the model limit.
        # Check the strlen and last offset.
        # We need to know this, for default implementation of throwing error.
        offsets = encoding.offsets
        type_ids = encoding.type_ids
        attn_mask = encoding.attention_mask

        # Find the last offset by counting attn masks
        # and keeping the last non-zero offset end.
        index = 0  # index of longest
        type_id = 0  # track type_id of longest

        for n, attn in enumerate(attn_mask):
            if attn == 1:
                end = offsets[n][1]  # Index to end character from offset
                if end > index:  # Grab last non-zero end index (ensures increasing too)
                    type_id = type_ids[n]
                    index = end
        end_index = index  # longest last char index
        end_typeid = type_id  # longest type (query or text)

        # If last token offset is before the last char, then it was truncated
        return end_index < len(texts[end_typeid].strip())

    def smart_batching_collate_text_only(
        self, batch, truncate_input_tokens: Optional[int] = 0
    ):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.get_tokenized(
            texts, truncate_input_tokens=truncate_input_tokens
        )

        return tokenized

    @staticmethod
    def raise_truncation_error(max_len, truncation_needed_indexes):

        indexes = f"{', '.join(str(i) for i in truncation_needed_indexes)}."
        index_hint = (
            " for text at "
            f"{'index' if len(truncation_needed_indexes) == 1 else 'indexes'}: {indexes}"
        )
        error.log_raise(
            "<NLP08391926E>",
            ValueError(
                f"Token sequence length (+3 for separators) exceeds the "
                f"maximum sequence length for this model ({max_len})"
                f"{index_hint}"
            ),
        )

    def predict(
        self,
        sentences: List[List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        num_workers: int = 0,
        activation_fct=None,
        apply_softmax=False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        truncate_input_tokens: Optional[int] = 0,
    ) -> PredictResultTuple:
        """
        Performs predictions with the CrossEncoder on the given sentence pairs.

        Args:
            See overriden method for details.
            truncate_input_tokens: Optional[int] = 0 added for truncation

        Returns:
            Uses PredictResultTuple to add input_token_count
        """
        input_was_string = False
        if isinstance(
            sentences[0], str
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        collate_fn = partial(
            self.smart_batching_collate_text_only,
            truncate_input_tokens=truncate_input_tokens,
        )
        iterator = DataLoader(
            sentences,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            shuffle=False,
        )

        if activation_fct is None:
            activation_fct = self.default_activation_function

        max_len = self.tokenizer.model_max_length
        pred_scores = []
        input_token_count = 0
        row = -1
        truncation_needed_indexes = []
        with torch.no_grad():
            for features in iterator:
                # Sum the length of all encodings for all samples
                for encoding in features.encodings:
                    row += 1

                    # for mask in encoding.attention_mask:
                    input_token_count += sum(encoding.attention_mask)

                    if truncate_input_tokens == 0 or truncate_input_tokens > max_len:
                        # default (for zero or over max) is to error on truncation
                        if self._truncation_needed(encoding, sentences[row]):
                            truncation_needed_indexes.append(row)

                if truncation_needed_indexes:
                    self.raise_truncation_error(max_len, truncation_needed_indexes)

                # We cannot send offset_mapping to the model with features,
                # but we needed offset_mapping for other uses.
                if "offset_mapping" in features:
                    del features["offset_mapping"]

                for name in features:
                    features[name] = features[name].to(self._target_device)

                model_predictions = self.model(**features, return_dict=True)
                logits = activation_fct(model_predictions.logits)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray(
                [score.cpu().detach().float().item() for score in pred_scores]
            )

        if input_was_string:
            pred_scores = pred_scores[0]

        return PredictResultTuple(pred_scores, input_token_count)

    def rank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = False,
        batch_size: int = 32,
        show_progress_bar: bool = None,
        num_workers: int = 0,
        activation_fct=None,
        apply_softmax=False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        truncate_input_tokens: Optional[int] = 0,
    ) -> RerankResultTuple:
        """
        Performs ranking with the CrossEncoder on the given query and documents.

        Returns a sorted list with the document indices and scores.

        Args:
            See overridden method for argument description.
            truncate_input_tokens (int, optional): Added to support truncation.
        Returns:
            RerankResultTuple: Adds input_token_count to result
        """
        query_doc_pairs = [[query, doc] for doc in documents]
        scores, input_token_count = self.predict(
            query_doc_pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            num_workers=num_workers,
            activation_fct=activation_fct,
            apply_softmax=apply_softmax,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            truncate_input_tokens=truncate_input_tokens,
        )
        results = []
        for i, score in enumerate(scores):
            if return_documents:
                results.append({"corpus_id": i, "score": score, "text": documents[i]})
            else:
                results.append({"corpus_id": i, "score": score})

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return RerankResultTuple(results[:top_k], input_token_count)
