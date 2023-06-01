# First Party
from caikit.core.data_model import enums
from caikit.core.data_model import GeneratedResult

# Local
from tests import EXTENSION

# Grab a handle to the data model of this extension module
extension_dm = EXTENSION.data_model


generated_response = GeneratedResult(
    generated_token_count=2, text="foo bar", stop_reason=enums.StopReason.STOP_SEQUENCE
)

# Add an example of one of each new data model types that you've defined within your extension
# to the list below. These samples are used for verifying that your object is well-aligned with
# with caikit serialization interfaces. They will only be used if your extension enables
# protobuf serialization in its config.
#
# NOTE: You do not need to add any samples from other extensions or the caikit library; only
# new types explicitly created in your extension data model.
data_model_samples = [
    generated_response,
]
