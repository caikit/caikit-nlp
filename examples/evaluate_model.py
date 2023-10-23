"""Given a trained model, which was presumably created by run_peft_tuning.py,
load it and run evaluation.
"""
# Standard
import argparse
import json
import pathlib

# Third Party
from tqdm import tqdm
from utils import (
    SUPPORTED_DATASETS,
    SUPPORTED_METRICS,
    configure_random_seed_and_logging,
    get_wrapped_evaluate_metric,
    is_float,
    kill_tgis_container_if_exists,
    load_model,
    print_colored,
    string_to_float,
)

# Local
from caikit_nlp.toolkit.verbalizer_utils import render_verbalizer


def parse_args() -> argparse.Namespace:
    """Parse & validate command line arguments.

    Returns:
        argparse.Namespace
            Parsed arguments to be leveraged model evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a text generation model.",
    )
    # TODO - Patch text-generation-launcher model var so that we can't mount the wrong model
    parser.add_argument(
        "--tgis",
        help="If enabled, runs inference through TGIS instead the local .run().",
        action="store_true",
    )
    parser.add_argument(
        "--model_path",
        help="Model to be loaded from disk.",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        help="Dataset to use to train prompt vectors. Options: {}".format(
            list(SUPPORTED_DATASETS.keys())
        ),
        default="twitter_complaints",
    )
    parser.add_argument(
        "--metrics",
        help="Metrics to calculate in space delimited list",
        default=["accuracy"],
        nargs="*",
        choices=list(SUPPORTED_METRICS.keys()),
    )
    parser.add_argument(
        "--preds_file",
        help="JSON file to dump raw source / target texts to.",
        default="model_preds.json",
    )
    parser.add_argument(
        "--max_new_tokens",
        help="Maximum number of new tokens to be generated",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--truncate_input_tokens",
        help="Number of allowed input tokens (no truncation=0)",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    return args


def get_model_preds_and_references(
    model, validation_stream, max_new_tokens, truncate_input_tokens
):
    """Given a model & a validation stream, run the model against every example in the validation
    stream and compare the outputs to the target/output sequence.

    Args:
        model
            Peft Model to be evaluated (may leverage different backends).
        validation_stream: DataStream[GenerationTrainRecord]
            Validation stream with labeled targets that we want to compare to our model's
            predictions.
        max_new_tokens: int
            Max number of new tokens to be generated, i.e., output limit
        truncate_input_tokens: int
            Number of allowed input tokens, i.e., input limit

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
            max_new_tokens=max_new_tokens,
            truncate_input_tokens=truncate_input_tokens,
        ).generated_text
        parse_pred_text = raw_model_text.split(datum.input)[-1].strip()
        model_preds.append(parse_pred_text)
        targets.append(datum.output)
    return (
        model_preds,
        targets,
    )


def export_model_preds(preds_file, predictions, validation_stream, verbalizer):
    """Exports a JSON file containing a list of objects, where every object contains:
        - source: str - Source string used for generation.
        - target: str - Ground truth target label used for generation.
        - verbalized_source: str - Source string after model verbalization
        - predicted_target: str - Predicted model target.

    Args:
        preds_file: str
            Path on disk to JSON file to be written.
        predictions: List
            Model prediction list, where each predicted text excludes source text as a prefix.
        validation_stream: DataStream
            Datastream object of GenerationTrainRecord objects used for validation against a model
            to generate predictions.
        verbalizer: str
            Model verbalizer used for generating target predictions.
    """
    pred_objs = []
    for pred, record in zip(predictions, validation_stream):
        res = {
            "source": record.input,
            "target": record.output,
            "predicted_target": pred,
        }
        if verbalizer is not None:
            res["verbalized_source"] = render_verbalizer(verbalizer, record)
        pred_objs.append(res)

    with open(preds_file, "w") as jfile:
        json.dump(pred_objs, jfile, indent=4, sort_keys=True)


if __name__ == "__main__":
    configure_random_seed_and_logging()
    args = parse_args()
    metric_funcs = [SUPPORTED_METRICS[metric] for metric in args.metrics]
    print_colored("Metrics to be calculated: {}".format(args.metrics))

    # Load the model; this can be a local model, or a distributed TGIS instance
    print_colored("Loading the model...")
    model = load_model(args.tgis, str(args.model_path))
    # Load the validation stream with marked target sequences
    print_colored("Grabbing validation data...")
    dataset_info = SUPPORTED_DATASETS[args.dataset]
    validation_stream = dataset_info.dataset_loader()[1]
    if validation_stream is None:
        raise ValueError(
            "Selected dataset does not have a validation dataset available!"
        )

    # Run the data through the model; save the predictions & references
    print_colored("Getting model predictions...")
    predictions, references = get_model_preds_and_references(
        model, validation_stream, args.max_new_tokens, args.truncate_input_tokens
    )
    print_colored(
        "Exporting model preds, source, verbalized source, and ground truth targets to {}".format(
            args.preds_file
        )
    )
    export_model_preds(
        args.preds_file,
        predictions,
        validation_stream,
        getattr(model, "verbalizer", None),
    )

    for metric_func in metric_funcs:
        metric_res = metric_func(predictions=predictions, references=references)
        print_colored(metric_res)
    # If we started a TGIS instance, kill it; otherwise, leave our container alone.
    # TODO: This will still looks for containers to kill, even if you're running TGIS
    # outside of a container through text-generation-server directly. For now, we are
    # always running TGIS in a container, so it's ok; the worst that will happen is
    # you'll kill somebody else's container.
    if args.tgis:
        print_colored("Killing containerized TGIS instance...")
        kill_tgis_container_if_exists()
