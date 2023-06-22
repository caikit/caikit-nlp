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
"""This module composes sentence splitting and sequence classification.
At this time this module is only designed for inference"""

# Standard
from typing import Dict, List
import os

# First Party
from caikit.core.modules import (
    ModuleBase,
    ModuleConfig,
    ModuleLoader,
    ModuleSaver,
    module,
)
from caikit.core.toolkit import error_handler
import alog

# Local
from ...data_model import TokenClassification, TokenClassificationResult
from ..sentence_split.base import SentenceSplitBase
from ..sequence_classification import SequenceClassification
from .token_classification_task import TokenClassificationTask

log = alog.use_channel("SEQ_CLASS")
error = error_handler.get(log)


@module(
    id="42a7d920-8b7e-4e1f-81fb-8ab851a80c99",
    name="Sequence transformer sentence-level classification",
    version="0.1.0",
    task=TokenClassificationTask,
)
class SequenceTransformerSentenceClassification(ModuleBase):

    ################################ Constructor #################################################

    def __init__(
        self,
        lang: str,
        sentence_splitter: SentenceSplitBase,
        sequence_classifier: SequenceClassification,
        labels_to_detect: List[str],
        labels_mapping: Dict[str, str] = None,
    ):
        """Construct a sentence content detection object from a
        sentence splitter and sequence classifier

        Args:
            lang: str
                2 letter language code
            sentence_splitter: SentenceSplitBase
                Sentence splitter
            sequence_classifier: SequenceClassification
                Sequence tokenizer and model
            labels_to_detect: List[str]
                Labels to detect
            labels_mapping: Dict[str, str]
                (Optional) Mapping of model labels to more semantically meaningful labels
        """
        super().__init__()
        error.type_check("<NLP12578168E>", str, lang=lang)
        error.type_check(
            "<NLP79642537E>", SentenceSplitBase, sentence_splitter=sentence_splitter
        )
        error.type_check(
            "<NLP35742128E>",
            SequenceClassification,
            sequence_classifier=sequence_classifier,
        )
        error.type_check_all("<NLP71653678E>", str, labels_to_detect=labels_to_detect)
        error.type_check("<NLP56932573E>", Dict, labels_mapping=labels_mapping)
        self.lang = lang
        self.sentence_splitter = sentence_splitter
        self.sequence_classifier = sequence_classifier
        self.labels_to_detect = labels_to_detect
        self.labels_mapping = labels_mapping

    ################################## API functions #############################################

    def run(self, text: str, threshold: float = 0.5) -> TokenClassificationResult:
        """Run sentence-level content detection on text. Returns results
        based on score threshold for labels that are to be detected

        Args:
            text: str
                Document to run sentence content detection on

        Returns:
            TokenClassificationResult
        """
        # TODO: make (default) threshold a part of the model and threshold itself Optional
        token_classification_results = []
        # Split document into sentences
        sentence_span_list = self.sentence_splitter.run(text)
        # Run in sentence through the classifier and determine based
        # on threshold and labels_to_detect what results should be returned
        text_list = [span.text for span in sentence_span_list]
        classification_results = self.sequence_classifier.run_batch(text_list)
        for idx, classification_result in enumerate(classification_results):
            # Each classification prediction is list of classifications
            # for that particular text example
            for classification in classification_result.results:
                # Map labels to semantic labels if provided

                # NOTE: labels need to be specified as str for config
                label = str(classification.label)
                if self.labels_mapping:
                    # Use original classifier label if not found
                    label = self.labels_mapping.get(label, label)
                if label in self.labels_to_detect and classification.score >= threshold:
                    sentence_span = sentence_span_list[idx]
                    token_classification = TokenClassification(
                        start=sentence_span.start,
                        end=sentence_span.end,
                        word=sentence_span.text,
                        entity_group=label,
                        score=classification.score,
                    )
                    token_classification_results.append(token_classification)
        return TokenClassificationResult(results=token_classification_results)

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
            module_saver.save_module(self.sentence_splitter, "sentence_split")
            module_saver.save_module(
                self.sequence_classifier, "sequence_classification"
            )
            config_options = {
                "language": self.lang,
                "labels_mapping": self.labels_mapping,
                "labels_to_detect": self.labels_to_detect,
            }
            module_saver.update_config(config_options)

    @classmethod
    def load(cls, model_path: str) -> "SequenceTransformerSentenceClassification":
        """Load a sentence content detection model. Assumes the
        tokenizer and model are HuggingFace transformer-based and on
        the same model_path

        Args:
            model_path: str
                Path to the model to be loaded.

        Returns:
           SequenceTransformerSentenceClassification
                Instance of this class built from the on disk model.
        """
        config = ModuleConfig.load(os.path.abspath(model_path))
        loader = ModuleLoader(model_path)
        sentence_splitter = loader.load_module("sentence_split")
        # Module loader looks for config.yml so we load this directly,
        # pending a way to load HF models directly through config.json
        # - https://github.com/caikit/caikit/issues/236
        sequence_classification_path = os.path.join(
            model_path, config.module_paths["sequence_classification"]
        )
        sequence_classifier = SequenceClassification.load(sequence_classification_path)
        return cls(
            sentence_splitter=sentence_splitter,
            sequence_classifier=sequence_classifier,
            lang=config.language,
            labels_to_detect=config.labels_to_detect,
            labels_mapping=config.labels_mapping,
        )
