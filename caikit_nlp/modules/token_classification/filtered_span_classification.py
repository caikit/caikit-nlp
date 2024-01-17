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
from caikit.core.exceptions import error_handler
from caikit.core.modules import (
    ModuleBase,
    ModuleConfig,
    ModuleLoader,
    ModuleSaver,
    module,
)
from caikit.interfaces.nlp.data_model import (
    TokenClassificationResult,
    TokenClassificationResults,
    TokenClassificationStreamResult,
)
from caikit.interfaces.nlp.tasks import (
    TextClassificationTask,
    TokenClassificationTask,
    TokenizationTask,
)
import alog

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
                Tokenizer that returns TokenizationResults
            classifier: ModuleBase
                Classification model instance returning Classification or
                TokenClassificationResult output on .run
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
            TokenizationTask in type(tokenizer).tasks,
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
        classification_tasks = type(classifier).tasks
        tasks_intersection = [i for i in classification_tasks if i in ALLOWED_TASKS]
        error.value_check(
            "<NLP41319814E>",
            any(tasks_intersection),
            f"classifier does not implement one of required tasks: {ALLOWED_TASKS}",
        )
        error.value_check(
            "<NLP41319815E>",
            len(tasks_intersection) == 1,
            f"classifier should implement only one task in: {ALLOWED_TASKS}",
        )
        self.lang = lang
        self.tokenizer = tokenizer
        self.classifier = classifier
        self.default_threshold = default_threshold
        self.labels_to_output = labels_to_output
        self.classification_task = tasks_intersection[0]

    ################################## API functions #############################################

    @TokenClassificationTask.taskmethod()
    def run(
        self, text: str, threshold: Optional[float] = None
    ) -> TokenClassificationResults:
        """Run classification on text split into spans. Returns results
        based on score threshold for labels that are to be outputted

        Args:
            text: str
                Document to run classification on
            threshold: float
                (Optional) Threshold based on which to return score results

        Returns:
            TokenClassificationResults
        """
        error.type_check("<NLP82129006E>", str, text=text)
        error.type_check("<NLP01414077E>", float, allow_none=True, threshold=threshold)

        if threshold is None:
            threshold = self.default_threshold
        if not text:
            # Allow empty text case to fall through - some tokenizers or
            # classifiers may error on this
            return TokenClassificationResults(results=[])

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
                        token_classification = TokenClassificationResult(
                            start=start,
                            end=end,
                            word=word,
                            entity=label,
                            score=classification.score,
                        )
                        token_classification_results.append(token_classification)
        return TokenClassificationResults(results=token_classification_results)

    @TokenClassificationTask.taskmethod(input_streaming=True, output_streaming=True)
    def run_bidi_stream(
        self, text_stream: Iterable[str], threshold: Optional[float] = None
    ) -> Iterable[TokenClassificationStreamResult]:
        """Run bi-directional streaming inferencing for this module.
        Run classification on text split into spans. Returns results
        based on score threshold for labels that are to be outputted

        Args:
            text_stream: Iterable[str]
                Text stream to run classification on
            threshold: float
                (Optional) Threshold based on which to return score results

        Returns:
            Iterable[TokenClassificationStreamResult]
        """
        error.type_check("<NLP96166348E>", float, allow_none=True, threshold=threshold)
        # TODO: For optimization implement window based approach.
        if threshold is None:
            threshold = self.default_threshold

        # Types on the stream are checked later on iteration
        if len(text_stream) == 0:
            # Allow empty text case to fall through - some tokenizers or
            # classifiers may error on this
            yield TokenClassificationStreamResult(results=[], processed_index=0)

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
                        yield TokenClassificationStreamResult(
                            results=[
                                TokenClassificationResult(
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
                yield TokenClassificationStreamResult(
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
                Tokenizer that returns TokenizationResults
            classifier: ModuleBase
                Classification model instance returning Classification or
                TokenClassificationResult output on .run
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
        span_start_offset = 0

        def __update_spans(token):
            # Check if the starting offset for the token is greater than
            # span_start_offset already, in which case, we need not
            # update the span
            if token.start < span_start_offset:
                # This is indicating that starting offset of token is off as we expect the token
                # to start at span_start_offset+1. So we need to recalibrate the token offsets
                # and have them start at span_start_offset. This means we need to know
                # the length of the token to manipulate the token.end, which we do by
                # subtracting end - start
                original_start = token.start
                token.start = span_start_offset
                token.end = span_start_offset + (token.end - original_start)
            return token

        for text in text_stream:
            error.type_check("<NLP38357927E>", str, text=text)
            stream_accumulator += text
            # In order to avoid processing all of the spans again, we only
            # send out the spans that are not yet finalized in detected_spans
            detected_spans = self.tokenizer.run(
                stream_accumulator[span_start_offset:]
            ).results

            if len(detected_spans) > 1:

                # we have detected more than 1 sentence
                # return 1st sentence
                new_span = detected_spans.pop(0)

                new_span = __update_spans(new_span)

                # we have detected new sentence, return the new sentence
                yield new_span

                # since we only send out part of the text, this means
                # the spans returned by the tokenizer will all now
                # start with last token offset + 1. So we store the last processed count
                # as the starting point for the subsequent one
                span_start_offset = new_span.end + 1

        # Return remaining sentence(s)
        if detected_spans and len(detected_spans) > 0:
            for detected_span in detected_spans:
                new_span = __update_spans(detected_span)
                yield new_span
