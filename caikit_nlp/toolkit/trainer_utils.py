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
"""Contains toolkit functionality for huggingface Trainer"""
# Standard
from datetime import datetime

# Third Party
import torch

# First Party
from caikit import get_config
from caikit.core.data_model import DataStream
from caikit.core.exceptions import error_handler
import alog

log = alog.use_channel("TRNR_UTILS")
error = error_handler.get(log)


def validate_training_data(train_stream: DataStream, model_name: str, module_id: str):

    global_default = get_config().training_data_limit.__default__
    module_default = (
        get_config()
        .training_data_limit.get(module_id, {})
        .get("__default__", global_default)
    )

    max_num_examples = (
        get_config()
        .training_data_limit.get(module_id, {})
        .get(model_name, module_default)
    )

    if max_num_examples > 0:
        train_stream_size = len(train_stream)
        error.value_check(
            "<NLP77627434E>",
            train_stream_size <= max_num_examples,
            "Number of examples ({}) exceeds the maximum number of examples allowed "
            "({}) for this model",
            train_stream_size,
            max_num_examples,
        )


def log_step(state, logs):
    if state.epoch is not None:
        logs["epoch"] = round(state.epoch, 2)

    # Get Rank
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    if "loss" in logs:
        if state.epoch is not None:
            logs["epoch"] = round(state.epoch, 2)

        log.debug(
            "process rank: {} loss: {} step: {}".format(
                rank, float(logs["loss"]), state.global_step
            )
        )
        output = {
            "epoch": float(logs["epoch"]),
            "step": state.global_step,
            "value": float(logs["loss"]),
            "timestamp": datetime.isoformat(datetime.now()),
        }
        state.log_history.append(output)
    else:
        output = {**logs, **{"step": state.global_step}}
        state.log_history.append(output)

    return state
