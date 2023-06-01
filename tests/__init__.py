import os
import importlib

from scripts.development.locutils import EXTENSION_NAME, PROJECT_ROOT_DIR


data_model_dir = os.path.join(PROJECT_ROOT_DIR, EXTENSION_NAME, "data_model")


def get_project_metadata(metadata_attr):
    """Getter for project metadata; checks environment variables first, and falls back to
    locator utilities if no environment variable is set. In general, environment variables
    should be set if the project source code is not available or the locator utilities are
    not available on the test runner's PYTHONPATH.

    Args:
        metadata_attr: str
            Property that we want to fetch from the locator utils, allowing env var overrides.
    Returns
        str
            Resolved metadata key to be leveraged in the tests.
    """
    # Environment variables take priority
    env_val = os.getenv(metadata_attr)
    if env_val:
        return env_val

    # If we don't have that, try to grab it off of our locator utils
    try:
        loc_utils = importlib.import_module("scripts.development.locutils")
        try:
            # If things go well, the requested key is on our locutils, and we can grab it
            return getattr(loc_utils, metadata_attr)
        except AttributeError:
            # Able to find locator utils, unable to find this metadata key in exports
            raise AttributeError(
                "metadata_attr {} is not an attribute of locutils".format(metadata_attr)
            )
    # Failed dynamic import
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "HINT: Unable to import locutils; try adding your extension to your PYTHONPATH"
        )


########## Exports ################################################################################
# Locator utilities metadata that we actually tend to use
CONFIG_PATH = get_project_metadata("CONFIG_PATH")
EXTENSION_NAME = get_project_metadata("EXTENSION_NAME")

# Handle to the extension itself
EXTENSION = importlib.import_module(EXTENSION_NAME)
