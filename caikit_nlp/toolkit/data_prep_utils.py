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

"""Helper functions for preparing training data, e.g., preparing data formats.
"""
from copy import deepcopy
from caikit.core.data_model import DataStream
from ..data_model import GenerationTrainRecord
from ..resources.pretrained_model import (
    HFAutoCausalLM,
    HFAutoSeq2SeqLM
)


IGNORE_ID = -100

def build_tokenize_function(tokenizer, max_source_length, max_target_length, verbalizer, task_type):
    def tokenize_function_language_model(example: GenerationTrainRecord):
        # NOTE: Truncation size for source is kind of silly - source/target seqs are disconnected.
        # Seq2seq should have the same issues...
        source_ids = tokenizer(
            example.input, max_length=max_source_length, truncation=True
        )
        target_ids = tokenizer(
            example.output, max_length=max_target_length, truncation=True
        )
        source_ids["input_ids"] = source_ids.input_ids + target_ids.input_ids
        # Here, we need to yield and manipulate the attention mask to attend
        # to the input seq + the tokens we have seen so far...
        num_target_samples = len(target_ids.input_ids)

        def generator_func():
            for idx in range(num_target_samples):
                # This may not actually be needed, but for now we do it, since the underlying data may be
                # referenced in multiple places, and the data will be dynamically padded by the LM collator
                s = deepcopy(source_ids)
                s["attention_mask"] = (
                    s["attention_mask"]
                    + [1] * (idx + 1)
                    + [0] * (num_target_samples - idx - 1)
                )
                yield (s)

        return DataStream(generator_func)

    def tokenize_function_seq2seq(
        example: GenerationTrainRecord,
    ) -> "BatchEncoding":
        """Tokenization function to be used for seq2seq training; this function consumes a
        GenerationTrainRecord object and applies the verbalizer to it followed by
        the model tokenizer. Finally, we postprocess by ignoring pad tokens in the label IDs.

        Args:
            example: GenerationTrainRecord
                Training data model object to convert a form we can learn on.

        Returns:
            transformers.tokenization_utils_base.BatchEncoding
                encoded tokenization output corresponding to the input example.
        """
        # Render the verbalizer template with the attributes of this data model example
        source = render_verbalizer(verbalizer, example)

        targets = example.output
        model_inputs = tokenizer(
            source,
            max_length=max_source_length,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
        )

        labels = labels["input_ids"]

        labels = list(
            map(lambda x: IGNORE_ID if x == tokenizer.pad_token_id else x, labels)
        )
        model_inputs["labels"] = labels
        model_inputs["task_ids"] = 0
        return model_inputs
    if task_type == HFAutoCausalLM.TASK_TYPE:
        return (tokenize_function_language_model, True)
    elif task_type == HFAutoSeq2SeqLM.TASK_TYPE:
        return (tokenize_function_seq2seq, False)
    else:
        raise ValueError("Task unsupported")
