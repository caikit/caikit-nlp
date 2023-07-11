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


# Third Party
from torch.utils.data import IterableDataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
)

# First Party
from caikit.core.data_model import DataStream
from caikit.core.modules import ModuleBase, module
from caikit.core.toolkit import error_handler, wip_decorator
import alog

# Local
from ...data_model import GeneratedResult, GenerationTrainRecord
from ...toolkit.data_stream_wrapper import SimpleIterableStreamWrapper
from ...toolkit.data_type_utils import get_torch_dtype
from .text_generation_task import TextGenerationTask

log = alog.use_channel("TXT_GEN")
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

    def __init__(self, tokenizer, model):
        super().__init__()

        self.tokenizer = tokenizer
        # NOTE: self.model here can also be HF trainer. This is because
        # if we have just trained the model then the models weights might be
        # available in different devices (and configuration), depending on
        # how it was trained. For now (July 10, 2023), we are not trying to
        # extract the model out from trainer itself, since that would require
        # us to essentially save it or reconstruct it to do normal inferring.
        self.model = model

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
        shuffle: bool = True,
        lr: float = 2e-5,
        checkpoint_dir: str = "/tmp",  # Directory where model predictions and checkpoints will be written
    ):
        """
        # FIXME: Below is currently configured for Seq2Seq only
        """

        torch_dtype = get_torch_dtype(torch_dtype)

        ## Load base model
        if isinstance(base_model, str):
            model_config = AutoConfig.from_pretrained(base_model)

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
            log.debug("Bootstrapping base resource [%s]", base_model)
            base_model = resource_type.bootstrap(base_model, torch_dtype=torch_dtype)

        ## Generate data loader from stream
        training_dataset: IterableDataset = cls._preprocess_function(
            train_stream=train_stream,
            tokenizer=base_model.tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            shuffle=shuffle,
        )

        ## TODO: Fetch trainer from resource

        # TODO: Make this whole thing configurable by end-users, by optionally accepting `training_args`
        # as argument to this train function.
        # TODO: Remove all the default used below and make them all configurable
        training_args = Seq2SeqTrainingArguments(
            output_dir=checkpoint_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            # NOTE: We have disabled evaluation for now
            do_eval=False,
            # evaluation_strategy = "epoch",
            learning_rate=lr,
            weight_decay=0.01,
            save_total_limit=3,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=False,
            no_cuda=False,  # Default
            generation_max_length=max_target_length,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_accumulation_steps=accumulate_steps,
            eval_accumulation_steps=accumulate_steps,
            # eval_steps=1,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=base_model.tokenizer, model=base_model.model
        )

        trainer = Seq2SeqTrainer(
            base_model.model,
            training_args,
            train_dataset=training_dataset,
            data_collator=data_collator,
            tokenizer=base_model.tokenizer,
            # compute_metrics=compute_metrics,
        )

        # Start training via Trainer.train function
        trainer.train()
        # NOTE: By default the model would be available in different ways
        # depending on where and how it was trained. So we need to fetch the model
        # from the trainer depending on the training method, like fsdp, ddp etc.
        # For simplicity, currently we will use trainer as the model since it anyways
        # enable the `predict` function on it and has all the layers of the model
        # distributed already, so it will be most optimized to use trainer to
        # perform prediction at this stage.

        return cls(
            tokenizer=base_model.tokenizer,
            model=trainer,
        )

    # pylint: disable=unused-argument
    def run(
        self, text, preserve_input_text=False, max_new_tokens=20, min_new_tokens=0
    ) -> "GeneratedResult":
        """Run inference against the model running in TGIS.

        Args:
            text: str
                Source string to be encoded for generation.
            preserve_input_text: bool
                Whether or not the source string should be contained in the generated output,
                e.g., as a prefix.
            max_new_tokens: int
                The maximum numbers of tokens to generate.
                Default: 20
            min_new_tokens: int
                The minimum numbers of tokens to generate.
                Default: 0 - means no minimum
        Returns:
            GeneratedResult
                Generated text result
        """
        if isinstance(self.model, Trainer):
            # Apply the tokenizer to the sample text & move to correct device
            tok_tensors = self.tokenizer(text, return_tensors="pt")
            # NOTE: below function is prediction on trainer, for which we need to supply the actual underlying model as well
            # NOTE: We are using prediction_step instead of calling `self.model.generate` because this way HF Trainer
            # automatically handles device placement of the data and model. Since the model is with Trainer at this point
            # and thus the device placement be according to training strategy, its better to let Trainer handle the
            # evaluation / prediction
            # NOTE: Below statement requires merge of HF PR https://github.com/huggingface/transformers/pull/24759
            # and subsequent release of `transformers` and updating the lib version in `caikit-nlp`
            # TODO: Add support for passing extra arguments to prediction_step
            _, generated_tokens, _ = self.model.prediction_step(
                self.model.model,
                tok_tensors,
                prediction_loss_only=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
            )

            generated_text = self.tokenizer.batch_decode(
                generated_tokens.detach().cpu().numpy(), skip_special_tokens=True
            )

        else:
            raise NotImplementedError(
                "model prediction on pre-finetuned model currently not supported"
            )

        return GeneratedResult(text=generated_text)

    ################################## Private Functions ###########################################

    @staticmethod
    def _preprocess_function(
        train_stream: DataStream[GenerationTrainRecord],
        tokenizer: AutoTokenizer,
        max_source_length: int,
        max_target_length: int,
        shuffle: bool,
    ):
        """Pre-process each example to get it prepared for training."""

        # FIXME: Below is currently configured for Seq2Seq only

        def _tokenization_func(
            example: GenerationTrainRecord,
        ):
            model_inputs = tokenizer(
                example.input,
                max_length=max_source_length,
                truncation=True,
            )

            labels = tokenizer(
                example.output,
                max_length=max_target_length,
                padding="max_length",
                truncation=True,
            )

            model_inputs["labels"] = labels["input_ids"]

            return model_inputs

        return SimpleIterableStreamWrapper(
            train_stream.map(_tokenization_func), shuffle=shuffle
        )
