"""This script illustrates how to fine-tune a model.

Supported model types:
- Seq2Seq LM
"""

# Standard
from typing import Any, Tuple
import argparse
import os
import shutil

# Third Party
from transformers import AutoConfig
from utils import (
    ALOG_OPTS,
    SUPPORTED_DATASETS,
    DatasetInfo,
    configure_random_seed_and_logging,
    print_colored,
)

# First Party
from caikit.core.data_model import DataStream
from caikit.core.toolkit import wip_decorator
import alog

# Local
from caikit_nlp.data_model import GenerationTrainRecord, TuningConfig
from caikit_nlp.modules.text_generation import FineTuning
from caikit_nlp.resources.pretrained_model import (
    HFAutoCausalLM,
    HFAutoSeq2SeqLM,
    PretrainedModelBase,
)

# TODO: Remove me once fine-tuning is out of WIP
wip_decorator.disable_wip()


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
    """Parse command line arguments.

    Returns:
        argparse.Namespace
            Parsed arguments to be leveraged for fine tuning application.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tuning a text generation model.",
    )
    # Register all of the common args, as well as specific tuning args for subcommands
    register_common_arguments(parser)

    args = parser.parse_args()
    # Reconfigure logging level based on verbosity, while preserving filters etc.

    alog_settings = {**ALOG_OPTS, **{"default_level": "debug"}}
    alog.configure(**alog_settings)
    # Validate common arg values
    validate_common_args(args)
    return args


def register_common_arguments(subparser: argparse.ArgumentParser) -> None:
    """Registers common arguments intended to be shared across all subparsers.

    Args:
        subparser: argparse.ArgumentParser
            Iterable of argument subparsers that should have common args.
    """
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
        default="t5-small",
    )
    subparser.add_argument(
        "--output_dir",
        help="Name of the directory that we want to export the model to",
        default="sample_tuned_model",
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


def show_experiment_configuration(args, dataset_info, model_type) -> None:
    """Show the complete configuration for this experiment, i.e., the model info,
    the resource type we built, the training params, metadata about the dataset where
    possible, and so on.

    Args:
        args: argparse.Namespace
            Parsed args corresponding to one tuning task.
        dataset_info: DatasetInfo
            Dataset information.
        model_type: type
            Resource class corresponding to the base model.
    """
    print_strs = [
        "Experiment Configuration",
        "- Model Name: [{}]".format(args.model_name),
        " |- Inferred Model Resource Type: [{}]".format(model_type),
        "- Dataset: [{}]".format(args.dataset),
        "- Number of Epochs: [{}]".format(args.num_epochs),
        "- Learning Rate: [{}]".format(args.learning_rate),
        "- Batch Size: [{}]".format(args.batch_size),
        "- Output Directory: [{}]".format(args.output_dir),
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

    # Init the resource & Build the tuning config from our dataset/arg info
    print_colored("[Loading the base model resource...]")
    base_model = model_type.bootstrap(args.model_name, tokenizer_name=args.model_name)

    # Then actually train the model & save it
    print_colored("[Starting the training...]")
    model = FineTuning.train(
        base_model,
        train_stream,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        accumulate_steps=args.accumulate_steps,
        num_epochs=args.num_epochs,
    )

    # model.save(args.output_dir, save_base_model=not args.prompt_only)

    print_colored("[Training Complete]")

    # Prediction
    sample_text = "this is sample text"
    prediction_results = model.run(sample_text)

    print("Generated text: ", prediction_results)
