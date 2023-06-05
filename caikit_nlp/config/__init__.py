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

# Standard
from pathlib import Path

# First Party
import alog
import caikit

log = alog.use_channel("CONFIG_INIT")

# The name for an extension is simply the name of the directory containing its config dir.
extension_name = Path(__file__).parent.parent.name

MODEL_MANAGER = caikit.core.MODEL_MANAGER

extract = MODEL_MANAGER.extract
load = MODEL_MANAGER.load
resolve_and_load = MODEL_MANAGER.resolve_and_load
