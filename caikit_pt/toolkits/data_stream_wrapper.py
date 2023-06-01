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
from torch.utils.data import IterableDataset

# First Party
import alog
from caikit.core.toolkit import error_handler


log = alog.use_channel("PEFT_PROMPT")
error = error_handler.get(log)


class SimpleIterableStreamWrapper(IterableDataset):
    """DataStream wrapper as an iterable PyTorch dataset; we use this to add
    compatability with PyTorch data loaders.
    """

    def __init__(self, stream, shuffle, buffer_size=None):
        error.type_check("<FPT12855513E>", bool, shuffle=shuffle)
        error.type_check(
            "<FPT12813713E>", int, buffer_size=buffer_size, allow_none=True
        )
        self.stream = stream
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        # Load the whole data set in memory
        if self.shuffle and buffer_size is None:
            self.buffer_size = len(stream)
        log.debug("Shuffling enabled? {}".format(self.shuffle))
        log.debug("Shuffling buffer size: {}".format(self.buffer_size))

    def __iter__(self):
        if self.shuffle:
            log.debug4("Reshuffling training data!")
            return iter(self.stream.shuffle(self.buffer_size))
        return iter(self.stream)

    def __len__(self):
        return len(self.stream)
