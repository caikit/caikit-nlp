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
"""Tests for the TGISAutoFinder"""

# Standard
from contextlib import contextmanager
from typing import Optional
from unittest.mock import patch
import copy

# Third Party
import pytest

# First Party
from caikit.config.config import merge_configs
from caikit.core.model_manager import ModelManager
from caikit_tgis_backend import TGISBackend
import aconfig
import caikit

# Local
from caikit_nlp.model_management import tgis_auto_finder
from caikit_nlp.modules.text_generation.text_generation_tgis import TextGenerationTGIS

## Helpers #####################################################################

# Convenient aliases
LOCAL_INITIALIZER_NAME = tgis_auto_finder.TGISAutoFinder._LOCAL_INITIALIZER_NAME_KEY
TGIS_BACKEND_PRIORITY = tgis_auto_finder.TGISAutoFinder._TGIS_BACKEND_PRIORITY_KEY


def make_tgis_config(hostname: str):
    return {
        "type": TGISBackend.backend_type,
        "config": {
            "test_connections": False,
            "connection": {
                "hostname": hostname,
            },
        },
    }


@contextmanager
def temp_model_manager(
    auto_finder_config: Optional[dict] = None,
    backend_priority: Optional[list] = None,
    local_initializer_key: str = "default",
):
    global_config = copy.deepcopy(getattr(caikit.config.config, "_CONFIG"))
    if backend_priority is None:
        backend_priority = [make_tgis_config("foo.bar:123")]
    updated_config = merge_configs(
        global_config,
        {
            "model_management": {
                "finders": {
                    "default": {
                        "type": tgis_auto_finder.TGISAutoFinder.name,
                        "config": auto_finder_config or {},
                    },
                },
                "initializers": {
                    local_initializer_key: {
                        "type": "LOCAL",
                        "config": {
                            "backend_priority": backend_priority,
                        },
                    },
                },
            },
        },
    )
    with patch("caikit.core.model_manager.get_config", lambda: updated_config):
        mmgr = ModelManager()
        # NOTE: The TGISAutoFinder relies on searching for the TGISBackend in
        #   the global caikit.core.MODEL_MANAGER and can inadvertently trigger
        #   that global instance to set up an initializer that prefers TGIS over
        #   LOCAL. We need isolation for tests, so we mock that global with this
        #   temporary instance here.
        with patch.object(tgis_auto_finder, "MODEL_MANAGER", new=mmgr):
            mmgr.initialize_components()
            yield mmgr


## Tests #######################################################################


def test_auto_find_tgis_model_ok():
    """Test that a TGIS text-gen model can be auto-found"""
    with temp_model_manager() as mmgr:
        model = mmgr.load("flan-t5-xl")
        assert model
        assert isinstance(model, TextGenerationTGIS)


def test_auto_find_tgis_model_non_default_local_initializer():
    """Test that a TGIS text-gen model can be auto-found when the local
    initializer is not the default
    """
    init_name = "notdefault"
    with temp_model_manager(
        auto_finder_config={LOCAL_INITIALIZER_NAME: init_name},
        local_initializer_key=init_name,
    ) as mmgr:
        model = mmgr.load("flan-t5-xl", initializer=init_name)
        assert model
        assert isinstance(model, TextGenerationTGIS)


def test_auto_find_tgis_model_multiple_tgis_backends_use_first():
    """Test that a TGIS text-gen model can be auto-found when there are
    multiple TGIS backends configured and no explicit priority is given
    """
    with temp_model_manager(
        backend_priority=[
            make_tgis_config("foo.bar:1234"),
            make_tgis_config("baz.bat:4567"),
        ]
    ) as mmgr:
        tgis_be0 = mmgr.get_initializer("default").backends[0]
        tgis_be1 = mmgr.get_initializer("default").backends[1]
        with patch.object(tgis_be0, "get_connection") as get_con_mock0:
            with patch.object(tgis_be1, "get_connection") as get_con_mock1:
                model = mmgr.load("flan-t5-xl")
                assert model
                assert isinstance(model, TextGenerationTGIS)
                get_con_mock0.assert_called()
                get_con_mock1.assert_not_called()


def test_auto_find_tgis_model_multiple_tgis_backends_set_order():
    """Test that a TGIS text-gen model can be auto-found when there are
    multiple TGIS backends configured and priority is explicitly given
    """
    with temp_model_manager(
        backend_priority=[
            make_tgis_config("foo.bar:1234"),
            make_tgis_config("baz.bat:4567"),
        ],
        auto_finder_config={TGIS_BACKEND_PRIORITY: 1},
    ) as mmgr:
        tgis_be0 = mmgr.get_initializer("default").backends[0]
        tgis_be1 = mmgr.get_initializer("default").backends[1]
        with patch.object(tgis_be0, "get_connection") as get_con_mock0:
            with patch.object(tgis_be1, "get_connection") as get_con_mock1:
                model = mmgr.load("flan-t5-xl")
                assert model
                assert isinstance(model, TextGenerationTGIS)
                get_con_mock0.assert_not_called()
                get_con_mock1.assert_called()


def test_bad_config_args():
    """Make sure that all flavors of bad configuration args are handled with
    appropriate errors
    """
    # Bad initializer name type
    with pytest.raises(TypeError):
        tgis_auto_finder.TGISAutoFinder(
            aconfig.Config(
                {LOCAL_INITIALIZER_NAME: 123},
                override_env_vars=False,
            )
        )

    # Bad tgis_backend_priority type
    with pytest.raises(TypeError):
        tgis_auto_finder.TGISAutoFinder(
            aconfig.Config(
                {TGIS_BACKEND_PRIORITY: "1"},
                override_env_vars=False,
            )
        )

    # Invalid tgis_backend_priority index value
    with pytest.raises(ValueError):
        tgis_auto_finder.TGISAutoFinder(
            aconfig.Config(
                {TGIS_BACKEND_PRIORITY: 123},
                override_env_vars=False,
            )
        )

    # Invalid tgis_backend_priority index value
    with pytest.raises(ValueError):
        tgis_auto_finder.TGISAutoFinder(
            aconfig.Config(
                {TGIS_BACKEND_PRIORITY: 123},
                override_env_vars=False,
            )
        )

    # Non-TGIS tgis_backend_priority index value
    with pytest.raises(ValueError):
        tgis_auto_finder.TGISAutoFinder(
            aconfig.Config(
                {TGIS_BACKEND_PRIORITY: 0},
                override_env_vars=False,
            )
        )

    # No TGIS backend available
    with pytest.raises(ValueError):
        tgis_auto_finder.TGISAutoFinder(aconfig.Config({}))


def test_unsupported_tgis_model():
    """Test that attempting to use a TGIS connection for a model that isn't
    supported by the TGIS backend results in not finding the model
    """
    with temp_model_manager(
        backend_priority=[
            {
                "type": TGISBackend.backend_type,
                "config": {
                    "remote_models": {
                        "not-flan-t5-xl": {
                            "hostname": "foo.bar:123",
                        }
                    }
                },
            }
        ],
    ) as mmgr:
        finder = mmgr._finders["default"]
        assert isinstance(finder, tgis_auto_finder.TGISAutoFinder)
        assert finder.find_model("flan-t5-xl") is None


def test_bad_tgis_connection():
    """Test that attempting to use a TGIS connection that can't connect results
    in not finding the model
    """
    with temp_model_manager(
        backend_priority=[
            {
                "type": TGISBackend.backend_type,
                "config": {
                    "remote_models": {
                        "flan-t5-xl": {
                            "hostname": "foo.bar:123",
                        }
                    },
                    "test_connections": True,
                    "connect_timeout": 5,
                },
            }
        ],
    ) as mmgr:
        finder = mmgr._finders["default"]
        assert isinstance(finder, tgis_auto_finder.TGISAutoFinder)
        assert finder.find_model("flan-t5-xl") is None
