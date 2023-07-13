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
import gc
import os

# Third Party
from transformers import AutoConfig
import torch

# First Party
from caikit.core.modules import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.core.toolkit import error_handler
import alog

# Local
from ...data_model import GeneratedResult
from ...resources.pretrained_model import (
    HFAutoCausalLM,
    HFAutoSeq2SeqLM,
    PretrainedModelBase,
)
from .text_generation_task import TextGenerationTask

log = alog.use_channel("TXT_GEN")
error = error_handler.get(log)


# pylint: disable=too-many-lines,too-many-instance-attributes
@module(
    id="f9174353-4aaf-4162-bc1e-f12dcda15292",
    name="Text Generation",
    version="0.1.0",
    task=TextGenerationTask,
)
class TextGenerationLocal(ModuleBase):
    """Module to provide text generation capabilities"""

    supported_resources = [HFAutoCausalLM, HFAutoSeq2SeqLM]

    def __init__(
        self,
        base_model_name: str,
        base_model: PretrainedModelBase = None,
        eos_token: str = None,
    ):
        super().__init__()

        error.type_check("<NLP48137045E>", str, allow_none=True, eos_token=eos_token)
        self.base_model = base_model
        self.base_model_name = base_model_name

        # Set _model_loaded as False by default. This will only get set to True if
        # we enable the tgis_backend and we are able to fetch the client successfully.
        self._model_loaded = False
        self._eos_token = eos_token

    def __del__(self):
        del self.base_model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except AttributeError:
            pass

    @classmethod
    def bootstrap(cls, base_model_path: str):
        """Function to bootstrap a pre-trained transformers model and
        get a caikit text-generation 'model'.

        Args:
            base_model_path: str
                Path to transformers model
                NOTE: Model path needs to contain tokenizer as well
        Returns:
            caikit_nlp.blocks.text_generation.TextGeneration
                Object of TextGeneration class (model)
        """
        # pylint: disable=duplicate-code
        model_config = AutoConfig.from_pretrained(base_model_path)

        resource_type = None
        for resource in cls.supported_resources:
            if model_config.model_type in resource.SUPPORTED_MODEL_TYPES:
                resource_type = resource
                break

        if not resource_type:
            error(
                "<NLP61784225E>",
                "{} model type is not supported currently!".format(
                    model_config.model_type
                ),
            )
        log.debug("Bootstrapping base resource [%s]", base_model_path)
        base_model = resource_type.bootstrap(
            base_model_path, tokenizer_name=base_model_path
        )
        eos_token = base_model._tokenizer.eos_token or None
        return cls(
            base_model_path,
            base_model,
            eos_token=eos_token,
        )
    
    @classmethod
    def load(cls, model_path: str) -> "TextGeneration":
        """Function to load text-generation model

        Args:
            model_path: str
                Path to the model to be loaded.
        Returns:
            TextGeneration
                Instance of this class built from the on disk model.
        """

        config = ModuleConfig.load(model_path)
        base_model_path = config.get("artifact_path", "")
        base_model_path = os.path.join(model_path, base_model_path)
        error.dir_check("<NLP01983374E>", base_model_path)
        return cls.bootstrap(base_model_path)

    def save(self, model_path):
        """Save caikit model

        Args:
            model_path: str
                Folder to save text-generation caikit model
        """
        saver = ModuleSaver(
            self,
            model_path=model_path,
        )
        with saver:
            artifacts_dir = "artifacts"
            saver.update_config(
                {
                    "artifact_path": artifacts_dir,
                    "eos_token": self._eos_token,
                }
            )
            if self.base_model:
                # This will save both tokenizer and base model
                self.base_model.save(
                    model_path,
                    tokenizer_dirname=artifacts_dir,
                    base_model_dirname=artifacts_dir,
                )

    def run(self, text, preserve_input_text=False, repetition_penalty=2.5, length_penalty=1.0, early_stopping=True, num_beams=1, max_new_tokens=20, min_new_tokens=0, **kwargs):
        """Run inference against the model running in TGIS.

        Args:
            text: str
                Source string to be encoded for generation.
            preserve_input_text: bool
                Whether or not the source string should be contained in the generated output,
                e.g., as a prefix.
            repetition_penalty: float 
                The parameter for repetition penalty. 1.0 means no penalty.
                Default: 2.5 
            length_penalty: float
                Exponential penalty to the length that is used with beam-based generation.
                It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence.
                Since the score is the log likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.
                Default: 1.0. 
            early_stopping: bool
                Controls the stopping condition for beam-based methods, like beam-search.
                It accepts the following values: True, where the generation stops as soon as there are num_beams complete candidates;
                False, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates; "never", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).
            num_beams: int
                Number of beams for beam search. 1 means no beam search.
                Default: 1
            max_new_tokens: int
                The maximum numbers of tokens to generate.
                Default: 20
            min_new_tokens: int
                The minimum numbers of tokens to generate.
                Default: 0 - means no minimum
            kwargs:
                Any other parameters to pass to generate as specified in GenerationConfig.
                https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/text_generation#transformers.GenerationConfig
        Returns:
            GeneratedResult
                Generated text result produced by the model.
        """

        inputs = self.base_model.tokenizer(text, return_tensors="pt")
        generate_ids = self.base_model.model.generate(
            input_ids=inputs["input_ids"],
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            use_cache=True,
            **kwargs,
        )
        token_count = generate_ids.size(1) - 1
        preds = [
            self.base_model.tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for g in generate_ids
        ]
        if generate_ids[0][-1].item() == self._eos_token:
            stop_reason = 'EOS_TOKEN'
        elif generate_ids.size(1) - 1 == max_new_tokens:
            stop_reason = 'MAX_TOKENS'
        else:
            stop_reason = 'OTHER'
        if preserve_input_text:
            generated_text = text + ": " + preds[0]
        else:
            generated_text = preds[0]
        return GeneratedResult(generated_token_count=token_count, text=generated_text, stop_reason=stop_reason, producer_id=self.PRODUCER_ID)
