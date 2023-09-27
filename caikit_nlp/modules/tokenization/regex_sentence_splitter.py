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
import os
import re

# First Party
from caikit.core.exceptions import error_handler
from caikit.core.modules import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.interfaces.nlp.data_model import Token, TokenizationResults
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
