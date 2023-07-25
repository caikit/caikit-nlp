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
from typing import Any, Iterator, Optional

# Third Party
from torch.utils.data import IterableDataset, get_worker_info

# First Party
from caikit.core.toolkit import error_handler
from caikit.core.data_model import DataStream
import alog

log = alog.use_channel("STREAM_WRAP")
error = error_handler.get(log)


class SimpleIterableStreamWrapper(IterableDataset):
    """DataStream wrapper as an iterable PyTorch dataset; we use this to add
    compatability with PyTorch data loaders.
    """

    def __init__(self, stream: DataStream[Any], shuffle: bool, buffer_size: Optional[int]=None, seed: int=42):
        error.type_check("<NLP12855513E>", bool, shuffle=shuffle)
        error.type_check(
            "<NLP12813713E>", int, buffer_size=buffer_size, allow_none=True
        )
        error.type_check("<NLP15553711E>", int, seed=seed)
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

    def __iter__(self) -> Iterator[Any]:
        """Initialize a consumable iterator. If we have n workers, we handle the shuffle
        behaviors first, then return every nth element, forming a partition across the
        substreams produced by each iterator at the cost of having to skip items. If
        we don't configure workers, we simply return prior to partitioning.

        Returns:
            Iterator
                iterator pertaining to one worker or the full dataset.
        """
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

    def _get_shuffle_seed(self, worker_info: Optional["WorkerInfo"]) -> int:
        """Gets the current seed for this shuffle.

        Args:
            worker_info: Optional["torch.utils.data._utils.worker.WorkerInfo"]
                Torch dataloader worker or None.

        Returns:
            int
                the seed to be used while for the next shuffle on the
                encapsulated stream.
        """
        if worker_info is None:
            return self.seed + self.shuffles_completed
        return self.seed + worker_info.dataset.shuffles_completed

    def _increment_shuffle_seed(self, worker_info: Optional["WorkerInfo"]) -> None:
        """Increments the current seed to prepare for the next shuffle.
        IMPORTANT: we must use persistent loaders when shuffling across
        multiple workers! Otherwise the worker will be destroyed, and our
        shuffle counter will be lost, which will cause shuffle to look
        like it's not working.

        Args:
            worker_info: Optional["torch.utils.data._utils.worker.WorkerInfo"]
                Torch dataloader worker or None.
        """
        if worker_info is None:
            self.shuffles_completed += 1
        else:
            worker_info.dataset.shuffles_completed += 1

    def _get_stream_partition(self,
                              cycle_stream: DataStream[Any],
                              worker_id: int,
                              num_workers: int):
        """Generator for a subset of a wrapped datastream; here, we simply traverse a stream,
        which is assumed to be preshuffled, and yield the elements that align with the
        scheme 'worker n gets every nth entry' after shuffling. This ensures that each
        record in the stream is encountered at most once per epoch as long as shuffling
        is consistent across the different workers.

        Args:
            cycle_stream: DataStream[Any]
                datastream that we're trying to partition.
            worker_id: int
                ID of the current worker.
            num_workers: int
                Number of workers being used to load the dataset.
        """
        for idx, elem in enumerate(cycle_stream):
            if (idx - worker_id) % num_workers == 0:
                yield elem

    def __len__(self) -> int:
        """Gets the encapsulated stream length.
        
        Returns:
            int
                number of objects in the stream.
        """
        return len(self.stream)
