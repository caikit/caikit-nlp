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
"""This module returns span classification output by splitting
text into spans and returning classifications for each span. Span
classifications can be filtered by score threshold and label(s).
At this time this module is only designed for inference"""

# Standard
from typing import List, Optional
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
from ..text_classification.text_classification_task import TextClassificationTask
from .token_classification_task import TokenClassificationTask

log = alog.use_channel("FILT_SPAN")
error = error_handler.get(log)

# Tasks allowed for the module to be provided for classifier
ALLOWED_TASKS = [TextClassificationTask, TokenClassificationTask]


@module(
    id="42a7d920-8b7e-4e1f-81fb-8ab851a80c99",
    name="Filtered span classification",
    version="0.1.0",
    task=TokenClassificationTask,
)
class FilteredSpanClassification(ModuleBase):

    ################################ Constructor #################################################

    def __init__(
        self,
        lang: str,
        span_splitter: ModuleBase,
        classifier: ModuleBase,
        default_threshold: float,
        labels_to_output: List[str] = None,
    ):
        """Construct a filtered span classification object
        from a span splitter and sequence classifier

        Args:
            lang: str
                2 letter language code
            span_splitter: ModuleBase
                Span splitter that returns List[Span]
            classifier: ModuleBase
                Classification model instance returning Classification or
                TokenClassification output on .run
            default_threshold: float
                Default threshold for scores
            labels_to_output: List[str]
                (Optional) Select labels to output, if None all labels will be returned
        """
        super().__init__()
        error.type_check("<NLP12578168E>", str, lang=lang)
        error.type_check("<NLP79642537E>", ModuleBase, span_splitter=span_splitter)
        error.type_check(
            "<NLP35742128E>",
            ModuleBase,
            classifier=classifier,
        )
        error.type_check("<NLP63802045E>", float, default_threshold=default_threshold)
        error.type_check_all(
            "<NLP71653678E>", str, allow_none=True, labels_to_output=labels_to_output
        )
        classification_task = classifier.TASK_CLASS
        if classification_task not in ALLOWED_TASKS:
            log.error(
                "<NLP92989564E>",
                f"classifier does not implement one of required tasks: {ALLOWED_TASKS}",
            )
        self.lang = lang
        self.span_splitter = span_splitter
        self.classifier = classifier
        self.default_threshold = default_threshold
        self.labels_to_output = labels_to_output
        self.classification_task = classification_task

    ################################## API functions #############################################

    def run(
        self, text: str, threshold: Optional[float] = None
    ) -> TokenClassificationResult:
        """Run classification on text split into spans. Returns results
        based on score threshold for labels that are to be outputted

        Args:
            text: str
                Document to run classification on
            threshold: float
                (Optional) Threshold based on which to return score results

        Returns:
            TokenClassificationResult
        """
        if threshold is None:
            threshold = self.default_threshold
        token_classification_results = []
        if self.classification_task == TextClassificationTask:
            # Split document into spans
            span_list = self.span_splitter.run(text)
            text_list = [span.text for span in span_list]
        else:
            # TokenClassificationTask classifiers would hold span info
            # so we do not need to span split again
            text_list = [text]
        # Run each span through the classifier and determine based
        # on threshold and labels_to_output what results should be returned
        classification_results = self.classifier.run_batch(text_list)
        for idx, classification_result in enumerate(classification_results):
            # Each classification result is list of classifications
            # for that particular text example
            for classification in classification_result.results:
                if self.classification_task == TextClassificationTask:
                    label = classification.label
                    span = span_list[idx]
                    start = span.start
                    end = span.end
                    word = span.text
                else:
                    label = classification.entity
                    start = classification.start
                    end = classification.end
                    word = classification.word
                if classification.score >= threshold:
                    if not self.labels_to_output or (
                        self.labels_to_output and label in self.labels_to_output
                    ):
                        token_classification = TokenClassification(
                            start=start,
                            end=end,
                            word=word,
                            entity=label,
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
            module_saver.save_module(self.span_splitter, "span_split")
            module_saver.save_module(self.classifier, "sequence_classification")
            config_options = {
                "language": self.lang,
                "default_threshold": self.default_threshold,
                "labels_to_output": self.labels_to_output,
            }
            module_saver.update_config(config_options)

    @classmethod
    def load(cls, model_path: str) -> "FilteredSpanClassification":
        """Load a filtered span classification model.

        Args:
            model_path: str
                Path to the model to be loaded.

        Returns:
            FilteredSpanClassification
                Instance of this class built from the on disk model.
        """
        config = ModuleConfig.load(os.path.abspath(model_path))
        loader = ModuleLoader(model_path)
        span_splitter = loader.load_module("span_split")
        classifier = loader.load_module("sequence_classification")
        return cls(
            span_splitter=span_splitter,
            classifier=classifier,
            lang=config.language,
            default_threshold=config.default_threshold,
            labels_to_output=config.labels_to_output,
        )

    @classmethod
    def bootstrap(
        cls,
        lang: str,
        span_splitter: ModuleBase,
        classifier: ModuleBase,
        default_threshold: float,
        labels_to_output: List[str] = None,
    ) -> "FilteredSpanClassification":
        """Bootstrap a FilteredSpanClassification instance

        Args:
            lang: str
                2 letter language code
            span_splitter: ModuleBase
                Span splitter that returns List[Span]
            classifier: ModuleBase
                Classification model instance returning Classification or
                TokenClassification output on .run
            default_threshold: float
                Default threshold for scores
            labels_to_output: List[str]
                (Optional) Select labels to output, if None all labels will be returned
        """
        # Basically just wrap the constructor currently
        return cls(
            lang=lang,
            span_splitter=span_splitter,
            classifier=classifier,
            default_threshold=default_threshold,
            labels_to_output=labels_to_output,
        )
