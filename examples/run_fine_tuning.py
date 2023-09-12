"""This script illustrates how to fine-tune a model.

Supported model types:
- Seq2Seq LM
"""

# Standard
from typing import Any, Tuple
import argparse
import json
import os
import shutil

# Third Party
from tqdm import tqdm
from transformers import AutoConfig
from utils import (
    ALOG_OPTS,
    SUPPORTED_DATASETS,
    SUPPORTED_METRICS,
    DatasetInfo,
    configure_random_seed_and_logging,
    load_model,
    print_colored,
)

# First Party
from caikit.core.data_model import DataStream
from caikit.core.toolkit import wip_decorator
import alog

# Local
from caikit_nlp.data_model import GenerationTrainRecord, TuningConfig
from caikit_nlp.modules.text_generation import TextGeneration
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
        default=2e-5,
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
        "--evaluate",
        help="Enable evaluation on trained model",
        action="store_true",
    )
    subparser.add_argument(
        "--preds_file",
        help="JSON file to dump raw source / target texts to.",
        default="model_preds.json",
    )
    subparser.add_argument(
        "--torch_dtype",
        help="Torch dtype to use for training",
        type=str,
        default="float16",
    )
    subparser.add_argument(
        "--metrics",
        help="Metrics to calculate. Options: {}".format(list(SUPPORTED_METRICS.keys())),
        nargs="*",
        default=["accuracy"],
    )
    subparser.add_argument(
        "--tgis",
        help="Run inference using TGIS. NOTE: This involves saving and reloading model in TGIS container",
        action="store_true",
    )
    subparser.add_argument(
        "--iterable_dataset",
        help="Enable evaluation on trained model",
        action="store_true",
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
        "- Enable evaluation: [{}]".format(args.evaluate),
        "- Evaluation metrics: [{}]".format(args.metrics),
        "- Torch dtype to use for training: [{}]".format(args.torch_dtype),
        "- Using iterable dataset: [{}]".format(args.iterable_dataset),
    ]
    # Log and sleep for a few seconds in case people actually want to read this...
    print_colored("\n".join([print_str for print_str in print_strs if print_str]))


def get_model_preds_and_references(
    model, validation_stream, truncate_input_tokens, max_new_tokens
):
    """Given a model & a validation stream, run the model against every example in the validation
    stream and compare the outputs to the target/output sequence.

    Args:
        model
            Fine-tuned Model to be evaluated (may leverage different backends).
        validation_stream: DataStream[GenerationTrainRecord]
            Validation stream with labeled targets that we want to compare to our model's
            predictions.
        truncate_input_tokens: int
            maximum number of tokens to be accepted by the model and rest will be
            truncated.
    Returns:
        Tuple(List)
            Tuple of 2 lists; the model predictions and the expected output sequences.
    """
    model_preds = []
    targets = []

    for datum in tqdm(validation_stream):
        # Local .run() currently prepends the input text to the generated string;
        # Ensure that we're just splitting the first predicted token & beyond.
        raw_model_text = model.run(
            datum.input,
            truncate_input_tokens=truncate_input_tokens,
            max_new_tokens=max_new_tokens,
        ).generated_text
        parse_pred_text = raw_model_text.split(datum.input)[-1].strip()
        model_preds.append(parse_pred_text)
        targets.append(datum.output)
    return (
        model_preds,
        targets,
    )


def export_model_preds(preds_file, predictions, validation_stream):
    """Exports a JSON file containing a list of objects, where every object contains:
        - source: str - Source string used for generation.
        - target: str - Ground truth target label used for generation.
        - predicted_target: str - Predicted model target.

    Args:
        preds_file: str
            Path on disk to JSON file to be written.
        predictions: List
            Model prediction list, where each predicted text excludes source text as a prefix.
        validation_stream: DataStream
            Datastream object of GenerationTrainRecord objects used for validation against a model
            to generate predictions.
    """
    pred_objs = []
    for pred, record in zip(predictions, validation_stream):
        pred_objs.append(
            {
                "source": record.input,
                "target": record.output,
                "predicted_target": pred,
            }
        )
    with open(preds_file, "w") as jfile:
        json.dump(pred_objs, jfile, indent=4, sort_keys=True)


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
    base_model = model_type.bootstrap(
        args.model_name, tokenizer_name=args.model_name, torch_dtype=args.torch_dtype
    )

    # Then actually train the model & save it
    print_colored("[Starting the training...]")
    model = TextGeneration.train(
        base_model,
        train_stream,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        lr=args.learning_rate,
        torch_dtype=args.torch_dtype,
        batch_size=args.batch_size,
        accumulate_steps=args.accumulate_steps,
        num_epochs=args.num_epochs,
        use_iterable_dataset=args.iterable_dataset,
    )

    print_colored("[Training Complete]")

    # Prediction
    sample_text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
    prediction_results = model.run(sample_text)

    print("Generated text: ", prediction_results)

    # Saving model
    model.save(args.output_dir)

    if args.tgis:

        # Load model in TGIS
        # HACK: export args.output_dir as MODEL_NAME for TGIS
        # container to pick up automatically
        os.environ["MODEL_DIR"] = os.path.dirname(args.output_dir)
        os.environ["MODEL_NAME"] = os.path.join(
            "models", os.path.basename(args.output_dir), "artifacts"
        )

        loaded_model = load_model(is_distributed=True, model_path=args.output_dir)

    else:
        # Use trained model directly
        loaded_model = model

    ## Evaluation
    print_colored("[Starting Evaluation]")

    validation_stream = dataset_info.dataset_loader()[1]

    print_colored("Getting model predictions...")
    truncate_input_tokens = args.max_source_length + args.max_target_length
    predictions, references = get_model_preds_and_references(
        loaded_model, validation_stream, truncate_input_tokens, args.max_target_length
    )

    export_model_preds(args.preds_file, predictions, validation_stream)

    metric_funcs = [SUPPORTED_METRICS[metric_name] for metric_name in args.metrics]
    print_colored("Metrics to be calculated: {}".format(args.metrics))

    for metric_func in metric_funcs:
        metric_res = metric_func(predictions=predictions, references=references)
        print_colored(metric_res)
