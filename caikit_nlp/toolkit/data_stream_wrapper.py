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
"""This module contains utility classes for wrapping data streams as Torch datasets efficiently.
Caikit modules leveraging such wrappers can use them to internally leverage common approaches
and objects for training / evaluating PyTorch models built around DataStreams, e.g., PyTorch
DataLoaders, with minimal boilerplate.
"""

# Third Party
from torch.utils.data import IterableDataset, get_worker_info

# First Party
from caikit.core.toolkit import error_handler
import alog

log = alog.use_channel("PEFT_PROMPT")
error = error_handler.get(log)


class SimpleIterableStreamWrapper(IterableDataset):
    """DataStream wrapper as an iterable PyTorch dataset; we use this to add
    compatability with PyTorch data loaders.
    """

    def __init__(self, stream, shuffle, buffer_size=None, seed=42):
        error.type_check("<NLP12855513E>", bool, shuffle=shuffle)
        error.type_check(
            "<NLP12813713E>", int, buffer_size=buffer_size, allow_none=True
        )
        self.seed = seed
        self.shuffles_completed = 0
        self.stream = stream
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        # Load the whole data set in memory
        if self.shuffle and buffer_size is None:
            self.buffer_size = len(stream)
        log.debug("Shuffling enabled? {}".format(self.shuffle))
        log.debug("Shuffling buffer size: {}".format(self.buffer_size))

    def __iter__(self):
        worker_info = get_worker_info()
        if self.shuffle:
            # Get the next shuffle seed; we use the root seed + number of
            # shuffles completed so far to ensure that every worker will
            # shuffle the same way for each epoch.
            shuffle_seed = self._get_shuffle_seed(worker_info)
            log.debug(f"Reshuffling training data with seed: {shuffle_seed}")
            cycle_stream = self.stream.shuffle(self.buffer_size, seed=shuffle_seed)
            self._increment_shuffle_seed(worker_info)
        else:
            cycle_stream = self.stream
        # Once shuffling has been handled, consider workers; if we have multiple
        # then create a substream from the main cycle stream to form a partition.
        if worker_info is not None:
            cycle_stream = self._get_stream_partition(
                cycle_stream, worker_info.id, worker_info.num_workers
            )
        return iter(cycle_stream)

    def _get_shuffle_seed(self, worker_info):
        if worker_info is None:
            return self.seed + self.shuffles_completed
        return self.seed + worker_info.dataset.shuffles_completed

    def _increment_shuffle_seed(self, worker_info):
        if worker_info is None:
            self.shuffles_completed += 1
        else:
            worker_info.dataset.shuffles_completed += 1

    def _get_stream_partition(self, cycle_stream, worker_id, num_workers):
        for idx, elem in enumerate(cycle_stream):
            if (idx - worker_id) % num_workers == 0:
                yield (elem)

    def __len__(self):
        return len(self.stream)
