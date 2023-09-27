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
import alog

log = alog.use_channel("TRNR_UTILS")


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
