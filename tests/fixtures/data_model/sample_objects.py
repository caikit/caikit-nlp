# First Party
from caikit.interfaces.nlp.data_model import FinishReason, GeneratedTextResult

generated_response = GeneratedTextResult(
    generated_text="foo bar",
    generated_tokens=2,
    finish_reason=FinishReason.STOP_SEQUENCE,
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
