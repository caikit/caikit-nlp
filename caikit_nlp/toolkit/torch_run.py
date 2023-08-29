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

"""This toolkit utility contains functions to operate on torch distribution

NOTE: Content of this file are heavily influenced by torch/distributed/run.py

Ref: https://github.com/pytorch/pytorch/blob/main/torch/distributed/run.py
"""

# Standard
import os

# Third Party
from torch import cuda
from torch.distributed.launcher.api import LaunchConfig, Std
import torch.distributed as dist

# First Party
import alog

log = alog.use_channel("TRCH_RN")


def initialize_torch_distribution(world_size, rank=0, backend="gloo|nccl"):

    if dist.is_available():
        log.debug(
            "Initializing process group - backend %s, rank %d, world size %d",
            backend,
            rank,
            world_size,
        )
        dist.init_process_group(backend=backend, world_size=world_size, rank=rank)


def determine_local_world_size():
    """Function to automatically deduce the world size based on
    available processors.

    NOTE: This function will try to use ALL gpus accessible to it
    """

    if cuda.is_available():
        num_proc = cuda.device_count()
        log.info("Cuda devices available! Using %d devices.", num_proc)
        return num_proc
    # Fall back to using the OS cpu count
    # TODO: Callibrate this to some reasonable default...
    num_proc = os.cpu_count()
    log.info("Cuda devices NOT available! Using CPU %d processes.", num_proc)
    return num_proc


def get_torch_elastic_launch_config(
    master_addr: str,
    master_port: str,
    start_method: str = "spawn",
    max_restarts=3,
) -> LaunchConfig:

    # Constants; we assume everything executes on the same node
    min_nodes = 1
    max_nodes = 1
    rdzv_configs = {"rank": 0}

    nproc_per_node = determine_local_world_size()

    if "OMP_NUM_THREADS" not in os.environ and nproc_per_node > 1:
        omp_num_threads = 1
        log.warning(
            "\n*****************************************\n"
            "Setting OMP_NUM_THREADS environment variable for each process to be "
            "%s in default, to avoid your system being overloaded, "
            "please further tune the variable for optimal performance in "
            "your application as needed. \n"
            "*****************************************",
            omp_num_threads,
        )
        # This env variable will be passed down to the subprocesses
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    return LaunchConfig(
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        nproc_per_node=nproc_per_node,
        start_method=start_method,
        rdzv_backend="static",
        rdzv_endpoint=f"{master_addr}:{master_port}",
        rdzv_configs=rdzv_configs,
        tee=Std.ALL,
        max_restarts=max_restarts,
    )
