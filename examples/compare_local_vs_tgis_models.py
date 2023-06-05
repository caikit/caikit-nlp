# Utils imports should always come first, because they ensure caikit_nlp
# is added to the syspath if not running inside of a container.
# Standard
import json
import time

# Third Party
from datasets import load_dataset
from utils import SUPPORTED_DATASETS, load_model
import torch

# First Party
import caikit

# Local
from caikit_nlp import data_model
import caikit_nlp

NUM_SAMPLES_TO_RUN = 100

model_path = "prompt_prefixes/sample_prompt"
# Grab the test stream from the twitter dataset
test_stream = SUPPORTED_DATASETS["twitter_complaints"].dataset_loader()[2]

# Load the local block
local_model = load_model(is_distributed=False, model_path=model_path)
# Load the TGIS backed block; this will kill TGIS if a container is already running, then restart.
distributed_model = load_model(is_distributed=True, model_path=model_path)

preds = []
for datum in test_stream:
    dis_res = distributed_model.run(datum.input)
    local_res = local_model.run(datum.input)
    preds.append(
        {
            "input": datum.input,
            "output": datum.output,
            "local_prediction": local_res.text.split(":")[-1].strip(),
            "distributed_prediction": dis_res.text.split(":")[-1].strip(),
        }
    )
    if len(preds) >= NUM_SAMPLES_TO_RUN:
        break


with open("preds.json", "w") as f:
    json.dump(preds, f, sort_keys=True, indent=4)

num_matching = 0
num_local_correct = 0
num_distributed_correct = 0
num_mismatching = 0
for x in preds:
    if x["output"] == x["local_prediction"]:
        num_local_correct += 1

    if x["output"] == x["distributed_prediction"]:
        num_distributed_correct += 1

    if x["local_prediction"] == x["distributed_prediction"]:
        num_matching += 1
    else:
        num_mismatching += 1

print("----- Metrics -----")
print("Num correct [local block via PEFT]: {}".format(num_local_correct))
print("Num correct [distributed via TGIS]: {}".format(num_distributed_correct))
print("Num matching remote / local preds: {}".format(num_matching))
print("Num not matching remote / local preds: {}".format(num_mismatching))
