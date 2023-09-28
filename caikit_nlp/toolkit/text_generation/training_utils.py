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
"""Utility script that contains logic for training"""

# Standard
from typing import Union

# Third Party
from datasets import Dataset
from datasets import IterableDataset as TransformersIterableDataset
from transformers import AutoTokenizer

# First Party
from caikit.core.data_model import DataStream
from caikit.core.toolkit import error_handler
import alog

# Local
from ...data_model import GenerationTrainRecord
from ...resources.pretrained_model import PretrainedModelBase

log = alog.use_channel("TXTGEN_TRN_UTLS")
error = error_handler.get(log)

# Below list is taken from
# https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
ALLOWED_TRAINING_ARGS = {
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "max_grad_norm",
    "lr_scheduler_type",
    "warmup_ratio",
    "warmup_steps",
    "use_ipex",
    "disable_tqdm",
    "label_names",
    "optim",
    "optim_args",
    "group_by_length",
    "dataloader_pin_memory",
    "gradient_checkpointing",
    "full_determinism",
}


def preprocess_function(
    base_model: PretrainedModelBase,
    train_stream: DataStream[GenerationTrainRecord],
    tokenizer: AutoTokenizer,
    max_source_length: int,
    max_target_length: int,
    shuffle: bool,
    use_iterable_dataset: bool,
    random_seed: int,
):
    """Pre-process each example to get it prepared for training."""
    dataset_type = TransformersIterableDataset if use_iterable_dataset else Dataset
    log.debug("Loading dataset class: [%s]", dataset_type.__name__)
    fn_kwargs = {
        "tokenizer": tokenizer,
        "max_source_length": max_source_length,
        "max_target_length": max_target_length,
    }
    # TODO: Add check for empty training stream
    dataset = dataset_type.from_generator(
        get_record, gen_kwargs={"train_stream": train_stream}
    )
    mapped_dataset = dataset.map(
        base_model.tokenize_function,
        fn_kwargs=fn_kwargs,
        batched=base_model.REQUIRES_TOKEN_UNWRAPPING,
        # Drop the input / output columns; we need to do this for dimensions to play
        # happily when operating on batched inputs for causal language modeling.
        remove_columns=["input", "output"],
    )

    if shuffle:
        log.debug("Shuffling the dataset")
        return mapped_dataset.shuffle(seed=random_seed)

    return mapped_dataset


def launch_training(
    base_model,
    training_dataset,
    training_args,
    checkpoint_dir,
    trainer=None,
    tokenizer=None,
) -> None:
    """Utility function to wrap trainer and execute training"""

    if not trainer:
        # If trainer is not provided fetch it from base_model
        if hasattr(base_model, "get_trainer"):
            trainer = base_model.get_trainer(
                train_dataset=training_dataset, **training_args
            )
        else:
            error("<NLP26155082E>", "could not resolve trainer. Check base model type!")

    # Start training via Trainer.train function
    trainer.train()

    # save the model temporarily and reload it
    # this is done, since otherwise the model might be distributed in different
    # devices, in which case its better to use trainer's `prediction_step`
    # functions, but then, they don't always give API similar to `generate`
    # and thus cause incompatibilities in `run` function
    trainer.save_state()
    trainer.save_model(checkpoint_dir)

    # save tokenizer explicitly
    if hasattr(base_model, "tokenizer"):
        base_model.tokenizer.save_pretrained(checkpoint_dir)
    elif tokenizer:
        tokenizer.save_pretrained(checkpoint_dir)
    else:
        log.warning("<NLP47068212W>", "tokenizer not available to train function.")

    # Below will return log history but launch will automatically attach rank to it.
    # if started in distributed fashion
    return trainer.state.log_history


def infer_max_steps(
    num_epochs: int,
    batch_size: int,
    training_dataset: Union[Dataset, TransformersIterableDataset],
):
    # Calculate the number of samples that we have
    if isinstance(training_dataset, Dataset):
        data_len = len(training_dataset)
    else:
        data_len = 0
        for _ in training_dataset:
            data_len += 1
    # Figure out how many batches we'll have per epoch
    num_batches = data_len // batch_size
    # Assume drop_last=False; in general, this doesn't really matter.
    # We mostly do this to avoid strange behavior when the dataset
    # size is smaller than the batch size.
    if num_batches != (data_len * batch_size):
        num_batches += 1
    num_steps = num_batches * num_epochs
    log.debug("Number of inferred steps: [%s]", num_steps)
    return num_steps


def get_record(train_stream):
    for data in train_stream:
        yield {"input": data.input, "output": data.output}