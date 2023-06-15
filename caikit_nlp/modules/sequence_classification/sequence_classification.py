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
"""This module contains sequence classification. At this time this module
is only designed for inference"""
# Standard
from typing import List, Union

# Third Party
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
import torch

# First Party
from caikit.core.modules import ModuleBase, module
from caikit.core.toolkit import error_handler
import alog

# Local
from ...data_model import TextClassification
from .text_classification_task import TextClassificationTask

log = alog.use_channel("SEQ_CLASS")
error = error_handler.get(log)


@module(
    id="d21107ca-d579-4321-aedd-3099a526e0dd",
    name="Sequence classification",
    version="0.1.0",
    task=TextClassificationTask,
)
class SequenceClassification(ModuleBase):

    ################################ Constructor #################################################

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        device: Union[str, int, None],
    ):
        super().__init__()
        error.type_check(
            "<NLP74125820E>",
            PreTrainedTokenizerBase,
            tokenizer=tokenizer,
        )
        error.type_check(
            "<NLP64751996E>",
            PreTrainedModel,
            model=model,
        )
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    ################################## API functions #############################################

    def run(self, text: str) -> TextClassification:
        """Run the sequence classification, truncates sequences too long for model

        Args:
            text: str
                Input string to be classified

        Returns:
            TextClassification
        """
        scores_dict = self._get_scores(text)
        # Re-organize scores_dict - for one text, this is just the first score
        classification = {
            label: score_array[0] for label, score_array in scores_dict.items()
        }
        return TextClassification(classification=classification)

    def run_batch(self, texts: List[str]) -> List[TextClassification]:
        """Run the sequence classification on batch, truncates sequences too long for model

        Args:
            text: List[str]
                Input strings to be classified

        Returns:
            List[TextClassification]
        """
        scores_dict = self._get_scores(texts)
        num_texts = len(texts)

        # Re-organize scores_dict for each example
        classifications = []
        for text_idx in range(num_texts):
            per_text_scores = {}
            for label, score_array in scores_dict.items():
                per_text_scores[label] = score_array[text_idx]
            classification = TextClassification(classification=per_text_scores)
            classifications.append(classification)
        return classifications

    def save(self, model_path: str):
        """Save model in target path

        Args:
            model_path: str
                Path to store model artifact(s)
            save_base_model: bool
                Save base model in the model_path provided.
                Default: False
        """
        self.tokenizer.save_pretrained(model_path)
        self.model.save_pretrained(model_path)

    @classmethod
    def load(cls, model_path: str) -> "SequenceClassification":
        """Load a tokenizer and sequence classification model. Assumes the
        tokenizer and model are HuggingFace transformer-based and on
        the same model_path

        Args:
            model_path: str
                Path to the model to be loaded.

        Returns:
            SequenceClassification
                Instance of this class built from the on disk model.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        device = SequenceClassification._get_device()
        return cls(
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

    ################################## Private Functions #########################################

    def _get_scores(self, text: Union[str, List[str]]):
        """Run tokenizer and model to get scores on text(s)

        Args:
            text: Union[str, List[str]]
                Input string(s) to be used

        Returns:
            scores_dict
                Dict with key label, and values as the array of scores,
                each corresponding to text(s)
        """
        # Apply tokenizer
        tokenized_text = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",  # PyTorch
        )
        # NOTE: no simple/efficient way to detect whether truncation
        # happened since padding also occurs, and this can be applied on
        # a batch of strings.
        if self.device == "cuda":
            tokenized_text = tokenized_text.to("cuda")
        with torch.no_grad():
            logits = self.model(**tokenized_text).logits

        if not self.model.config.id2label:
            log.warning(
                "<NLP31047577W>",
                "No id2label provided in model config. Defaulting to numeric labels",
            )

        softmax = torch.nn.Softmax(dim=1)
        raw_scores = softmax(logits)
        if self.device == "cuda":
            scores = raw_scores.cpu().numpy()
        else:
            scores = raw_scores.numpy()

        scores_dict = {}
        num_labels = self.model.num_labels
        for label_idx in range(num_labels):
            if self.model.config.id2label:
                label = self.model.config.id2label[label_idx]
            else:
                label = label_idx
            label_scores = scores[:, label_idx]
            scores_dict[label] = label_scores
        return scores_dict

    # NOTE: similar to prompt tuning but no user override, could consolidate eventually
    @staticmethod
    def _get_device() -> Union[str, int, None]:
        """Get the device which we expect to run our models on. Defaults to GPU
        if one is available, otherwise falls back to None (cpu).

        Returns:
            Union[str, int, None]
                Device string that we should move our models / tensors .to() at training
                and inference time.
        """
        device = "cuda" if torch.cuda.is_available() else None
        log.debug("Using device: %s", device)
        return device
