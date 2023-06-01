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
"""Setup to be able to build a wheel for this extension module.
"""

import os
import setuptools
import setuptools.command.build_py
from scripts.development.locutils import CONFIG_PATH, EXTENSION_NAME, PROJECT_ROOT_DIR

# We need to pass this in somehow by inferring it off of git tags, but we are leaving that for a separate PR
lib_version = os.getenv("COMPONENT_VERSION")
if lib_version is None:
    raise RuntimeError(
        "No version found; set the environment variable COMPONENT_VERSION"
    )


# read requirements from file
with open(os.path.join(PROJECT_ROOT_DIR, "requirements.txt")) as filehandle:
    requirements = list(map(str.strip, filehandle.read().splitlines()))
    # Remove --extra index line, as its not parsable by setup
    requirements = [
        name
        for name in requirements
        if not (name.startswith("--extra-index-url") or name.startswith("git"))
    ]

if __name__ == "__main__":

    setuptools.setup(
        name=EXTENSION_NAME,
        author="caikit",
        version=lib_version,
        license="Copyright Caikit Authors 2023 -- All rights reserved.",
        description="Foundation Model Prompt/Fine Tuning",
        install_requires=requirements,
        packages=setuptools.find_packages(include=("{}*".format(EXTENSION_NAME),)),
        data_files=[os.path.relpath(CONFIG_PATH)],
        include_package_data=True,
    )
