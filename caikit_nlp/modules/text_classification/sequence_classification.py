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
"""This module contains sequence classification, compatible with transformer
SequenceClassification modules. At this time this module is only designed
for inference"""
# Standard
from typing import Dict, List, Union

# Third Party
import torch

# First Party
from caikit.core.modules import ModuleBase, ModuleLoader, ModuleSaver, module
from caikit.core.toolkit import error_handler
import alog

# Local
from ...data_model import Classification, ClassificationResult
from ...resources.pretrained_model.hf_auto_seq_classifier import (
    HFAutoSequenceClassifier,
)
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
        resource: HFAutoSequenceClassifier,
        device: Union[str, int, None],
    ):
        super().__init__()
        error.type_check(
            "<NLP74125820E>",
            HFAutoSequenceClassifier,
            resource=resource,
        )
        self.resource = resource
        self.tokenizer = resource.tokenizer
        self.model = resource.model
        self.device = device

    ################################## API functions #############################################

    def run(self, text: str) -> ClassificationResult:
        """Run the sequence classification, truncates sequences too long for model

        Args:
            text: str
                Input string to be classified

        Returns:
            ClassificationResult
        """
        scores_dict = self._get_scores(text)
        # Re-organize scores_dict - for one text, this is just the first score
        return SequenceClassification._process_predictions(scores_dict, text_idx=0)

    def run_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """Run the sequence classification on batch, truncates sequences too long for model

        Args:
            text: List[str]
                Input strings to be classified

        Returns:
            List[ClassificationResult]
        """
        scores_dict = self._get_scores(texts)
        num_texts = len(texts)

        # Re-organize scores_dict for each example
        # We could eventually consider whether or not to sort classifications by scores
        # but avoiding this prescription here for now
        classification_predictions = []
        for text_idx in range(num_texts):
            classification_prediction = SequenceClassification._process_predictions(
                scores_dict, text_idx
            )
            classification_predictions.append(classification_prediction)
        return classification_predictions

    def save(self, model_path: str):
        """Save model in target path

        Args:
            model_path: str
                Path to store model artifact(s)
        """
        module_saver = ModuleSaver(
            self,
            model_path=model_path,
        )

        with module_saver:
            module_saver.save_module(self.resource, "sequence_classifier")

    @classmethod
    def load(cls, model_path: str) -> "SequenceClassification":
        """Load a sequence classification model

        Args:
            model_path: str
                Path to the model to be loaded.

        Returns:
            SequenceClassification
                Instance of this class built from the on disk model.
        """
        loader = ModuleLoader(model_path)
        resource = loader.load_module("sequence_classifier")
        device = SequenceClassification._get_device()
        return cls(resource=resource, device=device)

    @classmethod
    def bootstrap(cls, base_model_path: str) -> "SequenceClassification":
        """Bootstrap a HuggingFace transformer-based sequence classification model

        Args:
            base_model_path: str
                Path to the model to be loaded.
        """
        # Note: Must provide path to tokenizer if model_name is a path
        # for resource use
        resource = HFAutoSequenceClassifier.bootstrap(
            model_name=base_model_path, tokenizer_name=base_model_path
        )
        device = SequenceClassification._get_device()
        return cls(
            resource=resource,
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
            truncation=True,
            return_tensors="pt",  # PyTorch
        )
        # NOTE: no simple/efficient way to detect whether truncation
        # happened since padding also occurs, and this can be applied on
        # a batch of strings.
        if self.device == "cuda":
            tokenized_text = tokenized_text.to(self.device)
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

    @staticmethod
    def _process_predictions(scores_dict: Dict, text_idx: int) -> ClassificationResult:
        """Process dictionary of label: scores to ClassificationResult

        Args:
            scores_dict: Dict
                Dict with key label, and values as the array of scores,
                each corresponding to text(s)
            text_idx: int
                Integer index of text in batch

        Returns:
            ClassificationResult
        """
        error.type_check("<NLP40517898E>", Dict, scores_dict=scores_dict)
        classification_list = []
        for label, score_array in scores_dict.items():
            classification_list.append(
                Classification(label=label, score=score_array[text_idx])
            )
        return ClassificationResult(results=classification_list)

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
