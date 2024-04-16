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
"""Module that provides capability to split documents into sentences via regex"""

# Standard
from typing import Iterable
import itertools
import os
import re

# First Party
from caikit.core.exceptions import error_handler
from caikit.core.modules import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.interfaces.nlp.data_model import (
    Token,
    TokenizationResults,
    TokenizationStreamResult,
)
from caikit.interfaces.nlp.tasks import TokenizationTask
import alog

log = alog.use_channel("RGX_SNT_SPLT")
error = error_handler.get(log)


@module(
    id="1e04e21b-7009-499e-abdd-41e5984c2e7d",
    name="Regex Sentence Splitter",
    version="0.1.0",
    task=TokenizationTask,
)
class RegexSentenceSplitter(ModuleBase):
    # pylint: disable=anomalous-backslash-in-string
    """Use python regexes to split document into sentences.

    Sample regex string:
        [^.!?\s][^.!?\n]*(?:[.!?](?!['\"]?\s|$)[^.!?]*)*[.!?]?['\"]?(?=\s|$)
    """

    def __init__(self, regex_str: str):
        """Construct a RegexSentenceSplitter object
        by compiling the input regex string into python regex object
        that can be used later on for detection.

        Args:
            regex_str: str
                String containing pattern that can be complied with python re
                module
        """
        super().__init__()
        error.type_check("<NLP48846517E>", str, regex_str=regex_str)
        self.regex_str = regex_str
        self.regex = re.compile(self.regex_str)

    @classmethod
    def bootstrap(cls, regex_str) -> "RegexSentenceSplitter":
        """Bootstrap a a RegexSentenceSplitter object

        Args:
            regex_str: str
                String containing pattern that can be complied with python re
                module
        """
        return cls(regex_str)

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
            config_options = {"regex_str": self.regex_str}
            module_saver.update_config(config_options)

    @classmethod
    def load(cls, model_path: str) -> "RegexSentenceSplitter":
        """Load a regex sentence splitter model.

        Args:
            model_path: str
                Path to the model to be loaded.

        Returns:
            RegexSentenceSplitter
                Instance of this class built from the on disk model.
        """
        config = ModuleConfig.load(os.path.abspath(model_path))
        return cls(regex_str=config.regex_str)

    @TokenizationTask.taskmethod()
    def run(self, text: str) -> TokenizationResults:
        """Run sentence splitting regex on input text.

        Args:
            text: str
                Document to run sentence splitting on.

        Returns:
            TokenizationResults
                TokenizationResults object containing tokens where each token
                corresponds to a detected sentence.
        """

        error.type_check("<NLP38553904E>", str, text=text)

        matches = self.regex.finditer(text)
        tokens = []
        for match in matches:
            token = Token(start=match.start(), end=match.end(), text=match.group())
            tokens.append(token)

        return TokenizationResults(results=tokens)

    @TokenizationTask.taskmethod(input_streaming=True, output_streaming=True)
    def run_bidi_stream(
        self, text_stream: Iterable[str]
    ) -> Iterable[TokenizationStreamResult]:
        """Run bi-directional streaming sentence splitting. Aggregates text
        in the stream and returns back concatenable stream of sentences, with
        surrounding whitespace included

        Args:
            text_stream: Iterable[str]
                Text stream to run sentence splitting on

        Returns:
            Iterable[TokenizationStreamResult]
        """
        # Avoid length check here since it can be time consuming to iterate through stream
        # Tee stream to 2 - one to check emptiness, one for full iteration + analysis
        text_streams = itertools.tee(text_stream, 2)
        try:
            next(text_streams[0])
        except StopIteration:
            # Empty text case
            yield TokenizationStreamResult(results=[], start_index=0, processed_index=0)

        for token_output in self._stream_token_output(text_streams[1]):
            # start_index and processed_index here are simplified since each sentence
            # is expected to be concatenable and will be streamed
            yield TokenizationStreamResult(
                results=[token_output],
                start_index=token_output.start,
                processed_index=token_output.end,
            )

    ################################## Private functions ##########################################

    def _stream_token_output(self, text_stream):
        """Function to yield token output from input text stream"""
        # NOTE: Can potentially consolidate with parts of the filtered span classification function
        # in the future but this implementation currently works for tokens and accounts for
        # whitespace between sentences/tokens

        stream_accumulator = ""
        detected_tokens = None
        token_start_offset = 0
        len_missing_idx = 0
        # Tracker of text up until tokens/sentences detected - accounts for only whitespace case
        text_tracker = []

        def __update_tokens(token, stream_accumulator, len_missing_idx):
            # Check if the starting offset for the token is greater than
            # token_start_offset already, in which case, we need not
            # update the token
            if token.start < token_start_offset:
                # This is indicating that starting offset of sentence is off as we expect
                # the sentence to start at token_start_offset+1. So we need to recalibrate
                # the sentence offsets and have them start at token_start_offset. This
                # means we need to know the length of the sentence to manipulate the
                # token.end, which we do by subtracting end - start
                original_start = token.start
                token.start = token_start_offset
                token.end = (
                    token_start_offset + (token.end - original_start) + len_missing_idx
                )
                token.text = stream_accumulator[token.start : token.end]
            return token

        for text in text_stream:
            error.type_check("<NLP38367928E>", str, text=text)
            stream_accumulator += text
            text_tracker.append(text)

            # In order to avoid processing all of the tokens again, we only
            # send out the tokens that are not yet finalized in detected_tokens
            matches = self.regex.finditer(stream_accumulator[token_start_offset:])
            detected_tokens = []
            for match_token in matches:
                token = Token(
                    start=match_token.start(),
                    end=match_token.end(),
                    text=match_token.group(),
                )
                detected_tokens.append(token)

            if len(detected_tokens) > 1:
                # Optimization for not keeping track of all text chunks in the case
                # when there are actually sentences detected
                text_tracker = []

                # We have detected more than 1 sentence
                # Return 1st sentence
                new_token = detected_tokens.pop(0)

                new_token = __update_tokens(
                    new_token, stream_accumulator, len_missing_idx
                )

                # We have detected new sentence, return the new sentence
                yield new_token

                # We only send out part of the text, so we need to track
                # the starting point of the subsequent
                token_start_offset = new_token.end
                next_token_len = detected_tokens[0].end - detected_tokens[0].start
                len_missing_idx = (
                    len(stream_accumulator) - token_start_offset - next_token_len
                )

        # Return remaining sentence(s)
        if detected_tokens and len(detected_tokens) > 0:
            for detected_token in detected_tokens:
                new_token = __update_tokens(
                    detected_token, stream_accumulator, len_missing_idx
                )
                yield new_token

        else:
            # This allows us to keep track of text that is only whitespace that would
            # otherwise not return tokens since the tokenizer is only used to detect
            # sentences. This may have to be adjusted to keep track of any generated trailing
            # whitespace
            token_start = 0
            for text in text_tracker:
                token_end = token_start + len(text)
                yield Token(start=token_start, end=token_end, text=text)
                token_start = token_end
