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


def env_val_to_bool(val):
    """Returns the bool value of env var"""
    if val is None:
        return False
    if isinstance(val, bool):
        return val

    # For testing env vars for values that mean false (else True!)
    return str(val).lower().strip() not in ("no", "n", "false", "0", "f", "off", "")


def env_val_to_int(val, default):
    """Returns the integer value of env var or default value if None or invalid integer"""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default
