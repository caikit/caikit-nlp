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
from typing import Optional

# Third Party
from torch.utils.data import IterableDataset
from transformers import AutoConfig, AutoTokenizer
import torch

# First Party
from caikit.core.data_model import DataStream
from caikit.core.modules import ModuleBase, module
from caikit.core.toolkit import error_handler, wip_decorator
from caikit.interfaces.nlp.data_model import GeneratedTextResult
from caikit.interfaces.nlp.tasks import TextGenerationTask
import alog

# Local
from ...data_model import GenerationTrainRecord
from ...resources.pretrained_model import (
    HFAutoCausalLM,
    HFAutoSeq2SeqLM,
    PretrainedModelBase,
)
from ...toolkit.data_stream_wrapper import SimpleIterableStreamWrapper
from ...toolkit.data_type_utils import get_torch_dtype

log = alog.use_channel("FIN_TUN_GEN")
error = error_handler.get(log)


# pylint: disable=too-many-lines,too-many-instance-attributes
@module(
    id="28a81449-32ce-4be3-b688-545bde68f738",
    name="Text Generation",
    version="0.1.0",
    task=TextGenerationTask,
)
@wip_decorator.work_in_progress(
    category=wip_decorator.WipCategory.WIP, action=wip_decorator.Action.ERROR
)
class FineTuning(ModuleBase):
    """Module to provide fine-tuning support for text generation task"""

    supported_resources = [HFAutoCausalLM, HFAutoSeq2SeqLM]

    def __init__(
        self,
        tokenizer,
        model,
        bos_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.model = model
        self._bos_token = bos_token
        self._sep_token = sep_token
        self._eos_token = eos_token
        self._pad_token = pad_token

    @classmethod
    def train(
        cls,
        base_model: str,  # TODO: Union[str, PretrainedModelBase]
        train_stream: DataStream[GenerationTrainRecord],
        torch_dtype: str = None,  # TODO: Optional[Union[torch.dtype, str]]
        max_source_length: int = 256,
        max_target_length: int = 128,
        batch_size: int = 8,
        num_epochs: int = 5,
        accumulate_steps: int = 32,
        lr: float = 2e-5,
        # Directory where model predictions and checkpoints will be written
        checkpoint_dir: str = "/tmp",
        **training_arguments,
    ):
        """
        Fine-tune a CausalLM or Seq2seq text generation model.

        Args:
            base_model:  Union[str, caikit_nlp.resources.pretrained_model.base.PretrainedModelBase]
                Base resource model used for underlying generation.
            train_stream: DataStream[GenerationTrainRecord] or DataStream[ClassificationTrainRecord]
                Data to be used for training the prompt vectors of the generation model.
            torch_dtype: str
                TODO: Optional[Union[torch.dtype, str]]
                Data type to use for training/inference of the underlying text generation model.
                If no value is provided, we pull from torch_dtype in config. If an in memory
                resource is provided which does not match the specified data type, the model
                underpinning the resource will be converted in place to the correct torch dtype.
            max_source_length: int
                Max length of input sequences being considered. Default: 256.
            max_target_length: int
                Max length of target sequences being predicted. Default: 128.
            batch_size: int
                Batch sized to be used for training / evaluation data. Default: 8.
            num_epochs: int
                Number of epochs to tune the model. Default: 20.
            accumulate_steps: int
                Number of steps to use for gradient accumulation. Default: 1.
            lr: float
                Learning rate to be used while tuning model. Default: 2e-5.
            checkpoint_dir: str
                Directory where model predictions and checkpoints will be written
            **training_arguments:
                Arguments supported by HF Training Arguments.
                TrainingArguments:
                    https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/trainer#transformers.TrainingArguments
                Seq2SeqTrainingArguments:
                    https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments
        Returns:
            FineTuning
                Instance of this class with fine-tuned models.
        """

        torch_dtype = get_torch_dtype(torch_dtype)

        ## NOTE: Below code has been used in couple of places at this point, like in
        # text_generation module. In future, we would want to consolidate this into
        # a base class or a toolkit function
        # pylint: disable=duplicate-code
        resource_type = None

        ## Load base model
        if isinstance(base_model, str):
            model_config = AutoConfig.from_pretrained(base_model)

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
            log.debug("Bootstrapping base resource [%s]", base_model)
            base_model = resource_type.bootstrap(base_model, torch_dtype=torch_dtype)

        else:
            # base_model is actually a resource object
            resource_type = type(base_model)

        error.type_check("<NLP03221895E>", PretrainedModelBase, base_model=base_model)
        ## Generate data loader from stream
        training_dataset: IterableDataset = cls._preprocess_function(
            base_model=base_model,
            train_stream=train_stream,
            tokenizer=base_model.tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            shuffle=True,
        )

        ### Dtype based processing
        # NOTE: Following is not exhaustive list of all parameters
        # for all dtypes
        if torch_dtype == torch.float16:
            dtype_based_params = {
                "fp16": True,
            }
        elif torch_dtype == torch.bfloat16:
            dtype_based_params = {
                "bf16": True,
            }
        else:
            # default to float32
            dtype_based_params = {}

        ## TODO: Add automatic sharding selection based on number of parameters
        # in base model
        ## TODO: Fetch trainer from resource

        # TODO: Make this whole thing configurable by end-users,
        # by optionally accepting `training_args`
        # as argument to this train function.
        # TODO: Remove all the default used below and make them all configurable

        training_args = {
            "output_dir": checkpoint_dir,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "num_train_epochs": num_epochs,
            # NOTE: We have disabled evaluation for now
            "do_eval": False,
            # "evaluation_strategy ": "epoch",
            "learning_rate": lr,
            "weight_decay": 0.01,
            "save_total_limit": 3,
            "push_to_hub": False,
            "no_cuda": False,  # Default
            "remove_unused_columns": False,
            "dataloader_pin_memory": False,
            "gradient_accumulation_steps": accumulate_steps,
            "eval_accumulation_steps": accumulate_steps,
            # eval_steps=1,
            **training_arguments,
            **dtype_based_params,
        }

        trainer = base_model.get_trainer(
            train_dataset=training_dataset, **training_args
        )

        if num_epochs < 1:
            log.warning(
                "<NLP64076114W>",
                f"Number of epochs configured is {num_epochs} which is less than minimum 1. \
                    No training will be performed",
            )

            return cls(
                tokenizer=base_model.tokenizer,
                model=trainer,
            )

        # Start training via Trainer.train function
        trainer.train()

        # save the model temporarily and reload it
        # this is done, since otherwise the model might be distributed in different
        # devices, in which case its better to use trainer's `prediction_step`
        # functions, but then, they don't always give API similar to `generate`
        # and thus cause incompatibilities in `run` function
        trainer.save_model(checkpoint_dir)

        model = resource_type.bootstrap(
            checkpoint_dir, checkpoint_dir, torch_dtype=torch_dtype
        )

        return cls(
            tokenizer=model.tokenizer,
            model=model,
            bos_token=model.tokenizer.bos_token or None,
            sep_token=model.tokenizer.sep_token or None,
            eos_token=model.tokenizer.eos_token or None,
            pad_token=model.tokenizer.pad_token or None,
        )

    # pylint: disable=unused-argument
    def run(
        self, text, preserve_input_text=False, max_new_tokens=20, min_new_tokens=0
    ) -> "GeneratedTextResult":
        """Run inference against the model running in TGIS.

        Args:
            text: str
                Source string to be encoded for generation.
            preserve_input_text: bool
                Whether or not the source string should be contained in the generated output,
                e.g., as a prefix.
            max_new_tokens: int
                The maximum numbers of tokens to generate.
                Default: 128
            min_new_tokens: int
                The minimum numbers of tokens to generate.
                Default: 0 - means no minimum
        Returns:
            GeneratedTextResult
                Generated text result
        """

        inputs = self.model.tokenizer(text, return_tensors="pt")
        generate_ids = self.model.model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            use_cache=True,
        )

        token_count = generate_ids.size(1) - 1
        preds = [
            self.model.tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for g in generate_ids
        ]
        if generate_ids[0][-1].item() == self._eos_token:
            finish_reason = "EOS_TOKEN"
        elif generate_ids.size(1) - 1 == max_new_tokens:
            finish_reason = "MAX_TOKENS"
        else:
            finish_reason = "OTHER"

        return GeneratedTextResult(
            generated_tokens=token_count,
            generated_text=preds[0],
            finish_reason=finish_reason,
            producer_id=self.PRODUCER_ID,
        )

    ################################## Private Functions ###########################################

    @staticmethod
    def _preprocess_function(
        base_model: PretrainedModelBase,
        train_stream: DataStream[GenerationTrainRecord],
        tokenizer: AutoTokenizer,
        max_source_length: int,
        max_target_length: int,
        shuffle: bool,
    ):
        """Pre-process each example to get it prepared for training."""

        # TODO: We are using a default verbalizer which is strictly tied to
        # source training record currently. We need to figure out a better
        # way to make verbalizer optional for build_task_tokenize_function
        (
            tokenize_function,
            requires_unwrapping,
        ) = base_model.build_task_tokenize_function(
            tokenizer, max_source_length, max_target_length, verbalizer="{{input}}"
        )
        mapped_stream = train_stream.map(tokenize_function)
        if requires_unwrapping:
            mapped_stream = mapped_stream.flatten()

        return SimpleIterableStreamWrapper(mapped_stream, shuffle=shuffle)
