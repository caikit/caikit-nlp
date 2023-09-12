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
    DatasetInfo,
    load_json_file_dataset,
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
            list(SUPPORTED_DATASETS.keys()) + ["json_file"]
        ),
        default="twitter_complaints",
    )
    parser.add_argument(
        "--metrics",
        help="Metrics to calculate. Options: {}".format(list(SUPPORTED_METRICS.keys())),
        nargs="*",
        default=["accuracy"],
    )
    parser.add_argument(
        "--preds_file",
        help="JSON file to dump raw source / target texts to.",
        default="model_preds.json",
    )

    parser.add_argument(
        "--json_file_path",
        help="File path of the local JSON file to be loaded as dataset. It should be a JSON array or JSONL format.",
        default=None,
        type=pathlib.Path,
    )
    parser.add_argument(
        "--json_file_input_field",
        help="The input field to be used at the JSON file dataset",
        default="input",
    )
    parser.add_argument(
        "--json_file_output_field",
        help="The output field to be used at the JSON file dataset",
        default="output",
    )
    parser.add_argument(
        "--json_file_init_text",
        help="The init text to be used by the JSON file dataset",
        default="",
    )
    parser.add_argument(
        "--json_file_verbalizer",
        help="The custom verbalizer to be used by the JSON file dataset",
        default="{{input}}",
    )
    parser.add_argument(
        "--json_file_test_size",
        help="The percentage of the dataset that will be extracted as test set",
        default=0.1,
        type=float
    )
    parser.add_argument(
        "--json_file_validation_size",
        help="The percentage of the dataset that will be extracted as validation set",
        default=0.1,
        type=float
    )

    args = parser.parse_args()
    return args


def get_model_preds_and_references(model, validation_stream):
    """Given a model & a validation stream, run the model against every example in the validation
    stream and compare the outputs to the target/output sequence.

    Args:
        model
            Peft Model to be evaluated (may leverage different backends).
        validation_stream: DataStream[GenerationTrainRecord]
            Validation stream with labeled targets that we want to compare to our model's
            predictions.

    Returns:
        Tuple(List)
            Tuple of 2 lists; the model predictions and the expected output sequences.
    """
    model_preds = []
    targets = []

    for datum in tqdm(validation_stream):
        # Local .run() currently prepends the input text to the generated string;
        # Ensure that we're just splitting the first predicted token & beyond.
        if len(datum.input) > 0:
            raw_model_text = model.run(datum.input).text
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
        src, target = record.input, record.output
        pred_objs.append(
            {
                "source": record.input,
                "target": record.output,
                "predicted_target": pred,
                "verbalized_source": render_verbalizer(verbalizer, record),
            }
        )
    with open(preds_file, "w") as jfile:
        json.dump(pred_objs, jfile, indent=4, sort_keys=True)


if __name__ == "__main__":
    configure_random_seed_and_logging()
    args = parse_args()
    if not all(metric_name in SUPPORTED_METRICS for metric_name in args.metrics):
        raise KeyError(
            "One or more requested metrics {} not supported! Supported metrics: {}".format(
                args.metrics, list(SUPPORTED_METRICS.keys())
            )
        )
    metric_funcs = list(SUPPORTED_METRICS.values())
    print_colored("Metrics to be calculated: {}".format(args.metrics))
    # Load the model; this can be a local model, or a distributed TGIS instance
    print_colored("Loading the model...")
    model = load_model(args.tgis, str(args.model_path))
    # Load the validation stream with marked target sequences
    print_colored("Grabbing validation data...")
    if args.dataset != "json_file":
        dataset_info = SUPPORTED_DATASETS[args.dataset]
        validation_stream = dataset_info.dataset_loader()[1]
    else:
        dataset_info = DatasetInfo(
            verbalizer=args.json_file_verbalizer,
            dataset_loader=load_json_file_dataset(
                str(args.json_file_path),
                str(args.json_file_input_field),
                str(args.json_file_output_field),
                test_size=args.json_file_test_size,
                validation_size=args.json_file_validation_size
            ),
            init_text=args.json_file_init_text,
        )
        validation_stream = dataset_info.dataset_loader[1]

    if validation_stream is None:
        raise ValueError(
            "Selected dataset does not have a validation dataset available!"
        )

    # Run the data through the model; save the predictions & references
    print_colored("Getting model predictions...")
    predictions, references = get_model_preds_and_references(model, validation_stream)
    print_colored(
        "Exporting model preds, source, verbalized source, and ground truth targets to {}".format(
            args.preds_file
        )
    )
    export_model_preds(
        args.preds_file, predictions, validation_stream, model.verbalizer
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
