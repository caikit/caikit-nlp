"""This module holds references to common util functions and constants used for training
and evaluating models.
"""
# Standard
import os
import sys

# Hack for relative imports outside of containerized environments
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Standard
from collections import namedtuple
from shutil import which
from typing import Any, Callable, Tuple
import math
import random

# Third Party
import datasets
import evaluate
import numpy as np
import torch
import transformers

# First Party
from caikit_tgis_backend import TGISBackend
import alog
import caikit

# Local
from caikit_nlp.data_model import GenerationTrainRecord

# Silence noisy import time tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

ALOG_OPTS = {
    "filters": "datasets:off,urllib3:off,apscheduler:off,tzloc:off",
    "default_level": "error",
    "formatter": "pretty",
}

log = alog.use_channel("EXMPL_UTILS")


def configure_random_seed_and_logging():
    """Ensure that random experiments will be deterministic & set up default ALOG configuration."""
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    # Default Alog configuration; this may be overridden in tests as needed,
    # but some of these libraries have super verbose logging
    alog.configure(**ALOG_OPTS)


def print_colored(print_obj: Any) -> None:
    """Print stuff important to our experiment in blue text.
    Args:
        print_obj: Any
            Object to be printed.
    """
    print("\033[94m{}\033[0m".format(print_obj))


### Common loading / model capabilities
kill_tgis_container_if_exists = lambda: os.system("./kill-text-generation-launcher.sh")


def get_distributed_model(model_path):
    """Get the distributed implementation of the same model."""
    has_text_gen = which("text-generation-launcher")
    if not which("text-generation-launcher"):
        print("Text generation server command not found; using Docker override")
        this_dir = os.path.dirname(os.path.abspath(__file__))
        os.environ["PATH"] += ":" + this_dir
        assert (
            which("text-generation-launcher") is not None
        ), "Text generation script not found!"
        # Kill the docker container for TGIS if it's currently running
        kill_tgis_container_if_exists()

    # TODO: Enforce validation that TGIS is mounting the same model type

    caikit.config.configure(
        config_dict={
            "model_management": {
                "initializers": {
                    "default": {
                        "config": {
                            "backend_priority": [{"type": TGISBackend.backend_type}]
                        }
                    }
                }
            }
        }
    )  # should not be necessary but just in case
    dist_model = caikit.load(model_path)
    # Sanity check; if we have an environment variable override for the model TGIS is using,
    # make sure that its suffix (base model name) aligns with what we have in our config.
    # NOTE: bloom-560m is the default here because that's the default model used in our
    # text generation server hack script.
    model_name_override = os.getenv("MODEL_NAME", "bigscience/bloom-560m")
    if hasattr(dist_model, "base_model_name"):
        loaded_base_model = dist_model.base_model_name
    else:
        loaded_base_model = dist_model.model_name
    if not model_name_override.endswith(loaded_base_model):
        log.error(
            "TGIS using model name: {} conflicts with base model name: {}; set env var MODEL_NAME to the correct base model!".format(
                model_name_override, loaded_base_model
            )
        )

    return dist_model


def load_model(is_distributed: bool, model_path: str):
    """Loads a model for evaluation, either as a local module, or in a containerized
    instance of TGIS.

    """
    # Ensure caikit_nlp is locally imported, otherwise it'll be missing in out registry
    # Local
    import caikit_nlp

    # Validate that this model is something we actually know how to load
    if is_distributed:
        return get_distributed_model(model_path)
    return caikit_nlp.load(model_path)


### Dataset specific loader funcs
def load_twitter_dataset(
    get_test_set_as_eval=False,
) -> Tuple[caikit.core.data_model.DataStream]:
    """Load the ought/raft twitter complaints dataset.

    Returns:
        Tuple(caikit.core.data_model.DataStream)
            DataStreams of GenerationTrainRecords to be leveraged for training, validation,
            and testing, respectively.
    """
    to_generation_fmt = lambda x: GenerationTrainRecord(
        input=x["Tweet text"], output=x["text_label"]
    )
    build_stream = lambda split: caikit.core.data_model.DataStream.from_iterable(
        [to_generation_fmt(datum) for datum in dataset[split]]
    )
    dataset_for_task = datasets.load_dataset("ought/raft", "twitter_complaints")
    # Note: The data preprocessing (train, val and test) is copied from the Apache 2.0 code under
    #       https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_prompt_tuning_clm.ipynb
    classes = [k.replace("_", " ") for k in dataset_for_task["train"].features["Label"].names]  # type: ignore

    # Classes are ['Unlabeled', 'complaint', 'no complaint'] and given by integer => substitute int with NL label
    dataset = dataset_for_task.map(
        lambda x: {"text_label": [classes[label] for label in x["Label"]]},
        batched=True,
        num_proc=1,  # type: ignore
    )
    # This dataset has no validation data
    train_stream = build_stream("train")
    test_stream = build_stream("test")
    print_colored(
        "Warning: using train stream as validation; twitter data has no validation set!"
    )
    return (train_stream, train_stream, test_stream)


def load_cola_dataset() -> Tuple[caikit.core.data_model.DataStream]:
    """Load the Glue Cola dataset."""
    to_generation_fmt = lambda x: GenerationTrainRecord(
        input=x["sentence"], output=str(x["label"])
    )
    build_stream = lambda split: caikit.core.data_model.DataStream.from_iterable(
        [to_generation_fmt(datum) for datum in dataset[split]]
    )
    dataset = datasets.load_dataset("glue", "cola")
    train_stream = build_stream("train")
    validation_stream = build_stream("validation")
    test_stream = build_stream("test")
    return (train_stream, validation_stream, test_stream)


def load_rte_dataset() -> Tuple[caikit.core.data_model.DataStream]:
    """Load the Glue RTE dataset."""

    def to_generation_fmt(x):
        source_text = " ".join(
            ["sentence1:", x["sentence1"], "sentence2:", x["sentence2"]]
        )
        return GenerationTrainRecord(input=source_text, output=str(x["label"]))

    build_stream = lambda split: caikit.core.data_model.DataStream.from_iterable(
        [to_generation_fmt(datum) for datum in dataset[split]]
    )
    dataset = datasets.load_dataset("glue", "rte")
    train_stream = build_stream("train")
    validation_stream = build_stream("validation")
    test_stream = build_stream("test")
    return (train_stream, validation_stream, test_stream)


def load_mrpc_dataset() -> Tuple[caikit.core.data_model.DataStream]:
    """Load the Glue MRPC dataset."""

    def to_generation_fmt(x):
        source_text = " ".join(
            ["sentence1:", x["sentence1"], "sentence2:", x["sentence2"]]
        )
        return GenerationTrainRecord(input=source_text, output=str(x["label"]))

    build_stream = lambda split: caikit.core.data_model.DataStream.from_iterable(
        [to_generation_fmt(datum) for datum in dataset[split]]
    )
    dataset = datasets.load_dataset("glue", "mrpc")
    train_stream = build_stream("train")
    validation_stream = build_stream("validation")
    test_stream = build_stream("test")
    return (train_stream, validation_stream, test_stream)


def load_financial_phrasebank_dataset() -> Tuple[caikit.core.data_model.DataStream]:
    """Load the financial_phrasebank dataset."""

    def to_generation_fmt(x):
        return GenerationTrainRecord(input=x["sentence"], output=str(x["label"]))

    dataset = datasets.load_dataset("financial_phrasebank", "sentences_allagree")
    train_test_dataset = dataset["train"].train_test_split(test_size=0.1)
    # # Split the 10% test + valid into half test, half valid
    # test_valid = train_test_valid_dataset['test'].train_test_split(test=0.5)

    build_stream = lambda split: caikit.core.data_model.DataStream.from_iterable(
        [to_generation_fmt(datum) for datum in train_test_dataset[split]]
    )
    train_stream = build_stream("train")
    validation_stream = build_stream("test")
    test_stream = build_stream("test")
    return (train_stream, validation_stream, test_stream)


def load_billsum_dataset() -> Tuple[caikit.core.data_model.DataStream]:
    """Load the billsum dataset."""

    def to_generation_fmt(x):
        return GenerationTrainRecord(input=x["text"], output=str(x["summary"]))

    dataset = datasets.load_dataset("billsum", split="ca_test")
    train_test_dataset = dataset.train_test_split(test_size=0.2)
    # # Split the 10% test + valid into half test, half valid
    # test_valid = train_test_valid_dataset['test'].train_test_split(test=0.5)

    build_stream = lambda split: caikit.core.data_model.DataStream.from_iterable(
        [to_generation_fmt(datum) for datum in train_test_dataset[split]]
    )
    train_stream = build_stream("train")
    validation_stream = build_stream("test")
    test_stream = build_stream("test")
    return (train_stream, validation_stream, test_stream)


def load_samsum_dataset() -> Tuple[caikit.core.data_model.DataStream]:
    """Load the samsum dataset."""

    def to_generation_fmt(x):
        return GenerationTrainRecord(input=x["dialogue"], output=str(x["summary"]))

    dataset = datasets.load_dataset("samsum")
    train_test_dataset = dataset.train_test_split(test_size=0.1)
    # # Split the 10% test + valid into half test, half valid
    # test_valid = train_test_valid_dataset['test'].train_test_split(test=0.5)

    build_stream = lambda split: caikit.core.data_model.DataStream.from_iterable(
        [to_generation_fmt(datum) for datum in train_test_dataset[split]]
    )
    train_stream = build_stream("train")
    validation_stream = build_stream("validation")
    test_stream = build_stream("test")
    return (train_stream, validation_stream, test_stream)


def get_wrapped_evaluate_metric(metric_name: str, convert_to_numeric: bool) -> Callable:
    """Wrapper for running metrics out of evaluate which operate on numeric arrays
    named predictions & references, respectively.

    Args:
        metric_name: str
            Name of the metric to be run.
        convert_to_numeric: bool
            Indicates whether kwargs needed to be converted to floats prior to evaluation.
    Returns:
        Callable
            Callable evaluate metric which takes lists of strings as inputs.
    """
    metric = evaluate.load(metric_name)
    if convert_to_numeric:
        return build_func_for_numeric_input_metric_func(metric.compute)
    return metric.compute


def build_func_for_numeric_input_metric_func(func) -> Callable:
    """Builds a function that coerces its kwargs to numerics prior to forwarding to
    an internal function for metric evaluation.
    Args:
        func: Callable
            Function to be wrapped.

    Returns:
        Callable
            Function we can call with numeric inputs.
    """

    def metric_func_with_str_args(predictions: Tuple[str], references: Tuple[str]):
        # Convert all model predictions & targets to numerics
        numeric_preds = [string_to_float(pred, strict=False) for pred in predictions]
        numeric_refs = [string_to_float(ref, strict=True) for ref in references]
        return func(predictions=numeric_preds, references=numeric_refs)

    return metric_func_with_str_args


def is_float(string: str) -> bool:
    """Determine if a string is castable into a float.

    Args:
        string: str
            Value we want to try to cast to a float.

    Returns:
        bool
            True if the input string is castable to a bool, False otherwise.
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def string_to_float(string: str, strict: bool, default: int = -1.0):
    """Converts string to float, using default when conversion not possible.
    Here, we explicitly enforce that provided strings are numerically castable.

    Args:
        string: str
            Input string to be cast to numeric.
        default: int
            Default numeric label idx to map garbage

    Returns:
        int
            Label index of the cast result, which is presumably somehow mapped to classification
            (for currently supported metrics).
    """
    if strict and not is_float(string):
        raise ValueError(
            "Unable to cast string: [{}] to float in strict mode".format(string)
        )
    # Otherwise fall back to direct casting, and just return the cast value for
    # correctly generated indices, or the default value for what is presumably garbage.
    try:
        return float(string)
    except ValueError:
        return default


# Global map of supported datasets; users are able to select one of these to train prompt
# vectors against. Each value in the map should contain a default verbalizer, a nonparametric
# loader func for grabbing a tuple of three (train, validation||None, test||None) datastreams,
# and an initialization text string to use for initialize prompt tuning if TEXT is selected
# as the initialization option.
DatasetInfo = namedtuple("DatasetInfo", ["verbalizer", "dataset_loader", "init_text"])
SUPPORTED_DATASETS = {
    "twitter_complaints": DatasetInfo(
        verbalizer="Tweet text : {{input}} Label : ",
        dataset_loader=load_twitter_dataset,
        init_text="Classify if the tweet is a complaint or not:",
    ),
    "glue/cola": DatasetInfo(
        verbalizer="cola { 0 : grammatically unacceptable, 1 : grammatically acceptable } sentence: {{input}}",
        dataset_loader=load_cola_dataset,
        init_text="Classify if the text is a grammatical English sentence or not:",
    ),
    "glue/rte": DatasetInfo(
        verbalizer="rte { 0 : entailment, 1 : not entailment } {{input}}",
        dataset_loader=load_rte_dataset,
        init_text="Recognize textual entailment:",
    ),
    "glue/mrpc": DatasetInfo(
        verbalizer="mrpc { 0 : not equivalent, 1 : equivalent } {{input}}",
        dataset_loader=load_mrpc_dataset,
        init_text="Determine if the sentences are semantically equivalent: ",
    ),
    "financial_phrasebank": DatasetInfo(
        verbalizer="{{input}}",
        dataset_loader=load_financial_phrasebank_dataset,
        init_text="Classify sentiment for each of the news articles: ",
    ),
    "billsum": DatasetInfo(
        verbalizer="{{input}}",
        dataset_loader=load_billsum_dataset,
        init_text="",
    ),
    "samsum": DatasetInfo(
        verbalizer="{{input}}",
        dataset_loader=load_samsum_dataset,
        init_text="",
    ),
}

# Supported metrics in huggingface's evaluate library.
MetricInfo = namedtuple("MetricInfo", ["metric_name", "convert_to_numeric"])
METRIC_INFOS = [
    MetricInfo(metric_name="accuracy", convert_to_numeric=True),
    MetricInfo(metric_name="matthews_correlation", convert_to_numeric=True),
    MetricInfo(metric_name="rouge", convert_to_numeric=False),
]
SUPPORTED_METRICS = {
    metric_info.metric_name: get_wrapped_evaluate_metric(
        metric_info.metric_name, metric_info.convert_to_numeric
    )
    for metric_info in METRIC_INFOS
}
