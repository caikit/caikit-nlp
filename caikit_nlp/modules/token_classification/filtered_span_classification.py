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
from typing import Iterable, List, Optional
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
from ...data_model import (
    StreamingTokenClassificationResult,
    TokenClassification,
    TokenClassificationResult,
)
from ..text_classification.text_classification_task import TextClassificationTask
from .token_classification_task import TokenClassificationTask
from caikit_nlp.modules.tokenization.tokenization_task import TokenizationTask

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
        tokenizer: ModuleBase,
        classifier: ModuleBase,
        default_threshold: float,
        labels_to_output: List[str] = None,
    ):
        """Construct a filtered span classification object
        from a tokenizer and sequence classifier

        Args:
            lang: str
                2 letter language code
            tokenizer: ModuleBase
                Tokenizer that returns TokenizationResult
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
        error.type_check("<NLP79642537E>", ModuleBase, tokenizer=tokenizer)
        error.value_check(
            "<NLP42736791E>",
            tokenizer.TASK_CLASS == TokenizationTask,
            "tokenizer does not implement TokenizationTask",
        )
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
        error.value_check(
            "<NLP41319814E>",
            classification_task in ALLOWED_TASKS,
            f"classifier does not implement one of required tasks: {ALLOWED_TASKS}",
        )
        self.lang = lang
        self.tokenizer = tokenizer
        self.classifier = classifier
        self.default_threshold = default_threshold
        self.labels_to_output = labels_to_output
        self.classification_task = classification_task

    ################################## API functions #############################################

    @TokenClassificationTask.taskmethod()
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
            span_list = self.tokenizer.run(text).results
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

    @TokenClassificationTask.taskmethod(input_streaming=True, output_streaming=True)
    def run_bidi_stream(
        self, text_stream: Iterable[str], threshold: Optional[float] = None
    ) -> Iterable[StreamingTokenClassificationResult]:
        """Run bi-directional streaming inferencing for this module.
        Run classification on text split into spans. Returns results
        based on score threshold for labels that are to be outputted

        Args:
            text_stream: Iterable[str]
                Text stream to run classification on
            threshold: float
                (Optional) Threshold based on which to return score results

        Returns:
            Iterable[StreamingTokenClassificationResult]
        """
        # TODO: For optimization implement window based approach.
        if threshold is None:
            threshold = self.default_threshold

        for span_output in self._stream_span_output(text_stream):
            classification_result = self.classifier.run(span_output.text)
            results_to_end_of_span = False
            for classification in classification_result.results:
                if self.classification_task == TextClassificationTask:
                    label = classification.label
                    start = span_output.start
                    end = span_output.end
                    word = span_output.text
                else:
                    label = classification.entity
                    start = classification.start + span_output.start
                    end = classification.end + span_output.start
                    word = classification.word

                if classification.score >= threshold:
                    if not self.labels_to_output or (
                        self.labels_to_output and label in self.labels_to_output
                    ):
                        # Need to add offset to track actual place of spans within a stream,
                        # as the span splitting will be expected to stream and detect spans
                        yield StreamingTokenClassificationResult(
                            results=[
                                TokenClassification(
                                    start=start,
                                    end=end,
                                    word=word,
                                    entity=label,
                                    score=classification.score,
                                )
                            ],
                            processed_index=end,
                        )
                        if end == span_output.end:
                            results_to_end_of_span = True

            if not results_to_end_of_span:
                yield StreamingTokenClassificationResult(
                    results=[], processed_index=span_output.end
                )

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
            module_saver.save_module(self.tokenizer, "tokenizer")
            module_saver.save_module(self.classifier, "classification")
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
        tokenizer = loader.load_module("tokenizer")
        classifier = loader.load_module("classification")
        return cls(
            tokenizer=tokenizer,
            classifier=classifier,
            lang=config.language,
            default_threshold=config.default_threshold,
            labels_to_output=config.labels_to_output,
        )

    @classmethod
    def bootstrap(
        cls,
        lang: str,
        tokenizer: ModuleBase,
        classifier: ModuleBase,
        default_threshold: float,
        labels_to_output: List[str] = None,
    ) -> "FilteredSpanClassification":
        """Bootstrap a FilteredSpanClassification instance

        Args:
            lang: str
                2 letter language code
            tokenizer: ModuleBase
                Tokenizer that returns TokenizationResult
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
            tokenizer=tokenizer,
            classifier=classifier,
            default_threshold=default_threshold,
            labels_to_output=labels_to_output,
        )

    ################################## Private functions ##########################################

    def _stream_span_output(self, text_stream):
        """Function to yield span output from input text stream"""
        stream_accumulator = ""
        detected_spans = None
        detected_span_count = 0
        span_start_offset = 0

        def __update_spans(token):
            # Fix offset since we are restarting stream accumulator every-time.
            # (See comment below)
            token.start += span_start_offset
            token.end += span_start_offset
            return token

        for text in text_stream:
            stream_accumulator += text
            detected_spans = self.tokenizer.run(stream_accumulator).results

            if len(detected_spans) > 1:

                # we have detected more than 1 sentence
                # return 1st sentence
                new_span = detected_spans.pop(0)

                new_span = __update_spans(new_span)

                # we have detected new sentence, return the new sentence
                yield new_span

                detected_span_count += 1

                # Reset stream accumulator to optimize detection.
                # this way we don't have to keep tokenizing already tokenized
                # sentences
                stream_accumulator = text

                # since we are resetting the accumulator, this means
                # the spans returned by the tokenizer will all now
                # start with last token offset + 1. So we store the last processed count
                # as the starting point for the subsequent one
                span_start_offset = new_span.end + 1

        # For last remaining sentence
        if detected_spans and len(detected_spans) > 0:
            new_span = detected_spans.pop(0)
            new_span = __update_spans(new_span)
            yield new_span