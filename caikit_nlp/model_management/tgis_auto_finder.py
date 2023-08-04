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
"""
The TGISAutoFinder implements the ModelFinder interface to provide automatic
discovery of text-generation models that can be auto-configured to run against
a remote TGIS model.
"""
# Standard
from typing import Optional

# Third Party
import grpc

# First Party
from caikit import get_config
from caikit.core import MODEL_MANAGER, error_handler
from caikit.core.model_management import ModelFinderBase, model_finder_factory
from caikit.core.model_management.local_model_finder import LocalModelFinder
from caikit.core.modules import ModuleConfig
from caikit_tgis_backend import TGISBackend
from caikit_tgis_backend.protobufs import generation_pb2
import aconfig
import alog

# Local
from ..modules.text_generation import TextGenerationTGIS

log = alog.use_channel("TGIS_FND")
error = error_handler.get(log)


class TGISAutoFinder(ModelFinderBase):
    __doc__ = __doc__

    name = "TGIS-AUTO"

    # Constants for the keys of the config blob
    _LOCAL_INITIALIZER_NAME_KEY = "local_initializer_name"
    _TGIS_BACKEND_PRIORITY_KEY = "tgis_backend_priority"
    _TEST_CONNECTION_KEY = "test_connection"

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Initialize from the model finder factory config

        Config schema:

        local_initializer_name:
            type: string
            default: "default"
            description: The name within the initializers config for the LOCAL
                initializer that will hold the tgis backend to use

        tgis_backend_priority:
            type: integer
            description: Index within the backend_priority list for the TGIS
                backend to use. If not set, the first TGIS backend found will be
                used.

        test_connection:
            type: boolean
            description: If true, connections to TGIS remotes will be tested
                with a health probe before the model is considered found

        Args:
            config (aconfig.Config): The configuration blob from caikit's
                model_management factory construction
            instance_name (str): The name of this finder instance
        """
        local_initializer_name = config.get(self._LOCAL_INITIALIZER_NAME_KEY, "default")
        tgis_backend_priority = config.get(self._TGIS_BACKEND_PRIORITY_KEY)
        self._test_connection = config.get(self._TEST_CONNECTION_KEY, False)
        error.type_check(
            "<NLP97312902E>", str, local_initializer_name=local_initializer_name
        )
        error.type_check(
            "<NLP97312903E>",
            int,
            tgis_backend_priority=tgis_backend_priority,
            allow_none=True,
        )
        error.type_check("<NLP97312904E>", bool, test_connection=self._test_connection)

        # Extract the TGIS backend instance
        local_initializer = MODEL_MANAGER.get_initializer(local_initializer_name)
        backends = local_initializer.backends
        if tgis_backend_priority is not None:
            error.value_check(
                "<NLP87928813E>",
                tgis_back_priority >= 0 and tgis_backend_priority < len(backends),
                "Invalid {}: {}",
                self._TGIS_BACKEND_PRIORITY_KEY,
                tgis_backend_priority,
            )
            self._tgis_backend = backends[tgis_backend_priority]
            error.value_check(
                "<NLP77150201E>",
                tgis_backend.backend_type == TGISBackend.backend_type,
                "Index {} is not a TGIS backend",
                tgis_backend_priority,
            )
        else:
            tgis_backend = None
            for backend in backends:
                if backend.backend_type == TGISBackend.backend_type:
                    tgis_backend = backend
                    break
            error.value_check(
                "<NLP96294266E>",
                tgis_backend is not None,
                "No TGIS backend found!",
            )
            self._tgis_backend = tgis_backend

    def find_model(
        self,
        model_path: str,
        **kwargs,
    ) -> Optional[ModuleConfig]:
        """Find the model if"""

        # Get a connection to this model in tgis
        try:
            log.debug2("Attempting to setup TGIS client for %s", model_path)
            model_client = self._tgis_backend.get_client(model_id=model_path)
        except (TypeError, ValueError) as err:
            log.debug2("Unable to set up TGIS client for %s: %s", model_path, err)
            return None

        # If able to make a client, optionally test the connection
        if self._test_connection:
            log.debug2("Testing TGIS connection for %s", model_path)
            try:
                model_client.Tokenize(
                    generation_pb2.BatchedTokenizeRequest(
                        model_id=model_path,
                        requests=[generation_pb2.TokenizeRequest(text="test")],
                    )
                )
            except grpc.RpcError as err:
                log.debug2("Unable to make sample request to TGIS for %s", model_path)
                return None

        # If connection is ok, set up the module config to point to the remote
        # TGIS text generation module
        return ModuleConfig(
            {
                "module_id": TextGenerationTGIS.MODULE_ID,
                "module_class": TextGenerationTGIS.MODULE_CLASS,
                "name": TextGenerationTGIS.MODULE_NAME,
                "version": TextGenerationTGIS.MODULE_VERSION,
                "base_model_name": model_path,
            }
        )


model_finder_factory.register(TGISAutoFinder)
