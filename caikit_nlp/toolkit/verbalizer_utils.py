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
import re

# First Party
from caikit.core.exceptions import error_handler
import alog

log = alog.use_channel("VERBALIZER_UTIL")
error = error_handler.get(log)


def is_valid_verbalizer(verbalizer_template: str) -> bool:
    """Given a verbalizer template, determine if it's valid or not. We say a verbalizer
    template is valid if and only if we have at least one renderable field.

    Args:
        verbalizer_template: str
            Tentative verbalizer template to be used in text generation.
    Returns:
        bool:
            True if this is a valid verbalizer str with at least one renderable placeholder.
    """
    if not isinstance(verbalizer_template, str):
        return False
    return re.search(r"{{([_a-zA-Z0-9]+)}}", verbalizer_template) is not None


def render_verbalizer(verbalizer_template: str, source_object) -> str:
    """Given a verbalizer template and a source object, replace all templates with keys or
    attributes from the source object. Templates are expected to follow Python variable name
    allowed chars, i.e., alphanumeric with underscores; if they don't, they're skipped. The
    contents of the template will be replaced with either keys on the source object, or attribute
    values.

    Templates should be in double brackets with no whitespace, e.g., {{label}}. Consider the
    following examples.

    Examples:
        1. [Dictionary based]
            verbalizer_template = "Hello {{label}}"
            source_object = {"label": "world"}
            -> replace {{label}} with source_object["label"], producing "Hello world"
            returns: "Hello world"
        2. [Object based]
            verbalizer_template = "Source: {{input}} Target: {{output}}"
            source_object = GenerationTrainRecord(input="machine", output="learning")
            -> replace {{input}} with getattr(source_object, "source") and replace
               {{output}} with getattr(source_object, "target").
            returns: "Source: machine Target: learning"

    NOTE: This function will throw  KeyError/AttributeError if you try to grab a key or property
    that is invalid.

    Args:
        verbalizer_template: str
            Verbalizer that we want to render object values into.

    Returns:
        str
            Verbalizer string with placeholders rendered.
    """
    is_dict = isinstance(source_object, dict)

    def replace_text(match_obj):
        captured_groups = match_obj.groups()
        if len(captured_groups) != 1:
            error(
                "<NLP97444192E>",
                ValueError(
                    "Unexpectedly captured multiple groups in verbalizer rendering"
                ),
            )

        index_object = captured_groups[0]
        if is_dict:
            if index_object not in source_object:
                error(
                    "<NLP97415192E>",
                    KeyError("Requested template string is not a valid key in dict"),
                )
            return source_object[index_object]

        if not hasattr(source_object, index_object):
            error(
                "<NLP97715112E>",
                AttributeError(
                    "Requested template string is not a valid property of type"
                ),
            )
        return getattr(source_object, index_object)

    return re.sub(r"{{([_a-zA-Z0-9]+)}}", replace_text, verbalizer_template)
