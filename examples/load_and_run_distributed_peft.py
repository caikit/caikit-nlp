"""This script loads and runs a sample PEFT model as a caikit module using
the TGIS backend.

In a nutshell, this does the following:
- Check if `text-generation-launcher` is defined; if it doesn't, assume we have a Docker image
  that is (currently hardcoded to be) able to run TGIS & expose the proper ports, and patch a
  wrapper script around it onto our path so that the TGIS backend falls back to leveraging it
- Load the model through caikit
- Run an inference text generation request and dump the (garbage) output back to the console
"""
# Standard
from shutil import which
import os
import subprocess
import sys

# First Party
from caikit.core.module_backend_config import _CONFIGURED_BACKENDS, configure
from caikit_tgis_backend import TGISBackend
import alog
import caikit

# Local
import caikit_nlp

alog.configure("debug4")

PREFIX_PATH = "prompt_prefixes"

has_text_gen = which("text-generation-launcher")
if not which("text-generation-launcher"):
    print("Text generation server command not found; using Docker override")
    this_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PATH"] += ":" + this_dir
    assert (
        which("text-generation-launcher") is not None
    ), "Text generation script not found!"

# Configure caikit to prioritize TGIS backend
_CONFIGURED_BACKENDS.clear()
# load_timeout: 320
#           grpc_port: null
#           http_port: 3001
#           health_poll_delay: 1.0
caikit.configure(
    config_dict={"module_backends": {"priority": [TGISBackend.backend_type]}}
)  # should not be necessary but just in case
configure()  # backend configure

# Load with TGIS backend
prefix_model_path = os.path.join(PREFIX_PATH, "sample_prompt")
my_model = caikit.load(prefix_model_path)
sample_text = "@TommyHilfiger Dramatic shopping exp. ordered 6 jeans same size (30/32) 2 fits / 2 too large / 2 too slim : same brand &gt; different sizing"
sample_output = my_model.run(sample_text)

print("---------- Model result ----------")
print(sample_output)
