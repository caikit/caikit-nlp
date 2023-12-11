"""This script illustrates how to train a prompt tuned model using prompt tuning / MPT.
Supported tuning types:
- MultiPrompt Tuning
- Prompt Tuning

Supported model types:
- Causal LM
- Seq2Seq LM
"""
# Standard
from collections import namedtuple
from typing import Any, Tuple
import argparse
import os
import pathlib
import random
import shutil
import time

# Third Party
from peft.tuners.multitask_prompt_tuning import MultitaskPromptTuningInit
from peft.tuners.prompt_tuning import PromptTuningInit
from transformers import AutoConfig
from utils import (
    ALOG_OPTS,
    SUPPORTED_DATASETS,
    DatasetInfo,
    configure_random_seed_and_logging,
    print_colored,
)
import datasets

# First Party
from caikit.core.data_model import DataStream
import alog
import caikit

# Local
from caikit_nlp.data_model import GenerationTrainRecord, TuningConfig
from caikit_nlp.modules.text_generation.peft_prompt_tuning import (
    PeftPromptTuning,
    TuningType,
)
from caikit_nlp.resources.pretrained_model import (
    HFAutoCausalLM,
    HFAutoSeq2SeqLM,
    PretrainedModelBase,
)


def subsample_stream(
    train_stream: DataStream[GenerationTrainRecord], num_shots: int
) -> DataStream[GenerationTrainRecord]:
    """Given a training stream of length n, randomly extract num_shots <= n samples from it
    for use in few-shot learning.

    Args:
        train_stream: DataStream
            Full dataset to be used for training prior to few shot sampling.
        num_shots: int
            Number of samples to keep for use in training.

    Returns:
        DataStream[GenerationTrainRecord]
            Train subsampled stream of len(x) == num_shots
    """
    num_samples = len(train_stream)
    if num_shots > num_samples or num_shots <= 0:
        raise ValueError(
            "num_shots [{}] is less than 0 or exceeds train data size: [{}]".format(
                num_shots, num_samples
            )
        )
    # If we have the same number of samples as shots, just give the raw stream back
    elif num_shots == num_samples:
        return train_stream
    # Otherwise subsample the stream to condense its length; shuffle using
    # the whole stream as a buffer, and build a new stream from the result.
    # NOTE - this is not a great idea, but for now we do this, so that the sampling
    # is exactly the same as the original MPT code, since sampling the whole dataset
    # with a max buffer would load everything into memory anyway.
    shuffled_dataset = random.sample(list(train_stream), num_shots)
    return DataStream.from_iterable(shuffled_dataset)


def get_resource_type(model_name: str) -> PretrainedModelBase:
    """Given a model name, or a path to a model, get the resource type to be initialized.

    Args:
        model_name: str
            Model name or path to the model to be leveraged.

    Returns:
        type
            PretrainedModel subclass wrapping the loaded Transformer model, e.g.,
            a HFAutoCausalLM or HFAutoSeq2SeqLM. We return the type here so that
            we can initialize it later, after we show a nice experiment configuration.
    """
    try:
        model_type = AutoConfig.from_pretrained(model_name).model_type
    except OSError:
        raise ValueError("Failed to load model with name: {}".format(model_name))
    if model_type in HFAutoCausalLM.SUPPORTED_MODEL_TYPES:
        return HFAutoCausalLM
    elif model_type in HFAutoSeq2SeqLM.SUPPORTED_MODEL_TYPES:
        return HFAutoSeq2SeqLM
    raise NotImplementedError(
        "Provided is not supported for any supported resource type!"
    )


### Functions for arg parsing & validation
def parse_args() -> argparse.Namespace:
    """Parse command line arguments. Here, we set up each tuning task as a subparser
    to prevent the arguments from being too confusing. Common arguments, e.g., the number
    of virtual tokens, are added to all parsers.

    Returns:
        argparse.Namespace
            Parsed arguments to be leveraged for one prompt tuning application.
    """
    parser = argparse.ArgumentParser(
        description="Train prompt vectors on top of a text generation model.",
    )
    ### Args specific to subparsers, i.e., tuning / training arguments
    subparsers = parser.add_subparsers(
        help="The type of tuning to apply.", dest="tuning_type", required=True
    )
    # NOTE: These keys should line up with the TuningType enum values
    parser_multiprompt_tuning = subparsers.add_parser(
        "MULTITASK_PROMPT_TUNING",
        help="Train prompt vectors through Multitask Prompt Tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_prompt_tuning = subparsers.add_parser(
        "PROMPT_TUNING",
        help="Train prompt vectors through Prompt Tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = (
        parser_multiprompt_tuning,
        parser_prompt_tuning,
    )
    # Register all of the common args, as well as specific tuning args for subcommands
    register_common_arguments(subparsers)
    register_multitask_prompt_tuning_args(parser_multiprompt_tuning)
    register_prompt_tuning_args(parser_prompt_tuning)
    args = parser.parse_args()
    # Reconfigure logging level based on verbosity, while preserving filters etc.
    default_level = "debug" if args.verbose else "info"
    alog_settings = {**ALOG_OPTS, **{"default_level": default_level}}
    alog.configure(**alog_settings)
    # Validate common arg values
    validate_common_args(args)
    return args


def register_common_arguments(subparsers: Tuple[argparse.ArgumentParser]) -> None:
    """Registers common arguments intended to be shared across all subparsers.

    Args:
        subparsers: Tuple[argparse.ArgumentParser]
            Iterable of argument subparsers that should have common args.
    """
    for subparser in subparsers:
        subparser.add_argument(
            "--dataset",
            help="Dataset to use to train prompt vectors. Options: {}".format(
                list(SUPPORTED_DATASETS.keys())
            ),
            default="twitter_complaints",
        )
        subparser.add_argument(
            "--model_name",
            help="Name of base model or path to model to use to train prompt vectors",
            default="bigscience/bloom-560m",
        )
        subparser.add_argument(
            "--output_dir",
            help="Name of the directory that we want to export the model to",
            default="sample_prompt",
        )
        subparser.add_argument(
            "--prompt_only",
            help="Indicates that we do not need to export the full model, just the prompt vectors",
            action="store_true",
        )
        subparser.add_argument(
            "--verbose",
            help="If enabled, shows TQDM progress bars & debug logs",
            action="store_true",
        )
        subparser.add_argument(
            "--num_virtual_tokens",
            help="Number of virtual tokens to use per transformer submodule",
            type=int,
            default=8,
        )
        subparser.add_argument(
            "--num_epochs",
            help="Number of epochs to use for prompt tuning",
            type=int,
            default=10,
        )
        subparser.add_argument(
            "--learning_rate",
            help="Learning rate to use while training",
            type=float,
            default=3e-2,
        ),
        subparser.add_argument(
            "--num_shots",
            help="Number of training samples to use for few-shot learning",
            type=int,
            default=None,
        ),
        subparser.add_argument(
            "--batch_size", help="Batch size to use while training", type=int, default=8
        )
        subparser.add_argument(
            "--max_source_length",
            help="Maximum source sequence length.",
            default=256,
            type=int,
        )
        subparser.add_argument(
            "--max_target_length",
            help="Maximum target sequence length.",
            default=128,
            type=int,
        )
        subparser.add_argument(
            "--accumulate_steps",
            help="Gradient accumulation steps",
            default=1,
            type=int,
        )
        subparser.add_argument(
            "--torch_dtype",
            help="Torch dtype to be used for training",
            default="float32",
            choices=["float16", "bfloat16", "float32"],
        )


def register_multitask_prompt_tuning_args(subparser: argparse.ArgumentParser):
    """Register additional configuration options for MP(rompt)T subtask.

    Args:
        subparser: argparser.ArgumentParser
            Configuration options for multitask prompt tuning specifically.
    """
    subparser.add_argument(
        "--prompt_tuning_init",
        help="Initialization method to be used for multitask prompt tuning",
        choices=[x.value for x in MultitaskPromptTuningInit],
        default="TEXT",
    )
    subparser.add_argument(
        "--prompt_tuning_init_source_model",
        help="Path to source prompts to initialize multitask prompt tuning",
        type=pathlib.Path,
    )


def register_prompt_tuning_args(subparser: argparse.ArgumentParser):
    """Register additional configuration options for prompt tuning subtask.

    Args:
        subparser: argparser.ArgumentParser
            Configuration options for prompt tuning specifically.
    """
    subparser.add_argument(
        "--prompt_tuning_init",
        help="Initialization method to be used for prompt tuning",
        choices=[x.name for x in PromptTuningInit],
        default="TEXT",
    )


def validate_common_args(args: argparse.Namespace):
    """Validates common arguments to ensure values make sense; here, we only validate things that
    are not (or should not) be handled within the module.

    Args:
        args: argparse.Namespace
            Parsed args corresponding to one tuning task.
    """
    # Validate that the dataset is one of our allowed values
    if args.dataset not in SUPPORTED_DATASETS:
        raise KeyError(
            "[{}] is not a supported dataset; see --help for options.".format(
                args.dataset
            )
        )
    # Purge our output directory if one already exists.
    if os.path.isdir(args.output_dir):
        print("Existing model directory found; purging it now.")
        shutil.rmtree(args.output_dir)


def build_tuning_config(args: argparse.Namespace, dataset_info: DatasetInfo):
    """Builds the tuning config for this tuning task.

    Args:
        args: argparse.Namespace
            Parsed args corresponding to one tuning task.
        dataset_info: DatasetInfo
            Dataset information, including text to be used to initialize prompt tuning if
            that's the selected initialization scheme.

    Returns
        TuningConfig
            Tuning config object to be provided at .train() time.
    """
    # NOTE: the block itself already does filtering on our options, but since we separate things
    # through the CLI, we build the Tuning Config as accurately as possible so that the options
    # can be shown in the console.
    base_kwargs = {
        "num_virtual_tokens": args.num_virtual_tokens,
        "prompt_tuning_init_method": args.prompt_tuning_init,
    }
    # Add the initialization text only if we actually initialize with text
    if args.prompt_tuning_init == "TEXT":
        base_kwargs["prompt_tuning_init_text"] = dataset_info.init_text
    if (
        args.tuning_type == "MULTITASK_PROMPT_TUNING"
        and args.prompt_tuning_init_source_model
    ):
        if not args.prompt_tuning_init_source_model.exists():
            raise FileNotFoundError(
                "Provided prompt tuning init source does not exist!"
            )
        base_kwargs["prompt_tuning_init_source_model"] = str(
            args.prompt_tuning_init_source_model
        )
    return TuningConfig(**base_kwargs)


def show_experiment_configuration(args, dataset_info, model_type) -> None:
    """Show the complete configuration for this experiment, i.e., the model info,
    the resource type we built, the training params, metadata about the dataset where
    possible, and so on.

    Args:
        args: argparse.Namespace
            Parsed args corresponding to one tuning task.
        dataset_info: DatasetInfo
            Dataset information, including text to be used to initialize prompt tuning if
            that's the selected initialization scheme.
        model_type: type
            Resource class corresponding to the base model.
    """
    text_init_substr = (
        " |- Prompt Tuning initialization Text: [{}]".format(dataset_info.init_text)
        if args.prompt_tuning_init == "TEXT"
        else ""
    )
    print_strs = [
        "Experiment Configuration",
        "- Model Name: [{}]".format(args.model_name),
        " |- Inferred Model Resource Type: [{}]".format(model_type),
        "- Tuning Type: [{}]".format(args.tuning_type),
        "- Prompt Tuning Initialization Type [{}]".format(args.prompt_tuning_init),
        "- Number of Virtual Tokens: [{}]".format(args.num_virtual_tokens),
        text_init_substr,
        "- Dataset: [{}]".format(args.dataset),
        "- Verbalizer: [{}]".format(dataset_info.verbalizer),
        "- Number of Epochs: [{}]".format(args.num_epochs),
        "- Learning Rate: [{}]".format(args.learning_rate),
        "- Batch Size: [{}]".format(args.batch_size),
        "- Output Directory: [{}]".format(args.output_dir),
        "- Exporting prompt only: [{}]".format(args.prompt_only),
        "- Number of shots: [{}]".format(args.num_shots),
        "- Maximum source sequence length: [{}]".format(args.max_source_length),
        "- Maximum target sequence length: [{}]".format(args.max_target_length),
        "- Gradient accumulation steps: [{}]".format(args.accumulate_steps),
    ]
    # Log and sleep for a few seconds in case people actually want to read this...
    print_colored("\n".join([print_str for print_str in print_strs if print_str]))


if __name__ == "__main__":
    configure_random_seed_and_logging()
    args = parse_args()
    model_type = get_resource_type(args.model_name)
    # Unpack the dataset dictionary into a loaded dataset & verbalizer
    dataset_info = SUPPORTED_DATASETS[args.dataset]
    show_experiment_configuration(args, dataset_info, model_type)
    # Convert the loaded dataset to a stream
    print_colored("[Loading the dataset...]")
    # TODO - conditionally enable validation stream
    train_stream = dataset_info.dataset_loader()[0]
    if args.num_shots is not None:
        train_stream = subsample_stream(train_stream, args.num_shots)
    # Init the resource & Build the tuning config from our dataset/arg info
    print_colored("[Loading the base model resource...]")
    base_model = model_type.bootstrap(args.model_name, tokenizer_name=args.model_name)
    tuning_config = build_tuning_config(args, dataset_info)
    # Then actually train the model & save it
    print_colored("[Starting the training...]")
    model = PeftPromptTuning.train(
        base_model,
        train_stream,
        tuning_config,
        val_stream=None,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        tuning_type=args.tuning_type,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        verbalizer=dataset_info.verbalizer,
        silence_progress_bars=not args.verbose,
        accumulate_steps=args.accumulate_steps,
        torch_dtype=args.torch_dtype,
    )
    model.save(args.output_dir, save_base_model=not args.prompt_only)
    print_colored("[Training Complete]")
