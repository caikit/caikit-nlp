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
Tests for tgis_utils
"""

# Standard
from typing import Iterable, Optional, Type

# Third Party
import fastapi
import grpc
import grpc._channel
import pytest

# First Party
from caikit.core.data_model import ProducerId
from caikit.core.exceptions.caikit_core_exception import CaikitCoreException
from caikit.interfaces.runtime.data_model import RuntimeServerContextType
from caikit_tgis_backend.protobufs import generation_pb2

# Local
from caikit_nlp.toolkit.text_generation import tgis_utils
from tests.fixtures import TestServicerContext

## Helpers #####################################################################


class MockTgisClient:
    """Mock of a TGIS client that doesn't actually call anything"""

    def __init__(
        self,
        status_code: Optional[grpc.StatusCode],
        error_message: str = "Yikes",
    ):
        self._status_code = status_code
        self._error_message = error_message

    def _maybe_raise(self, error_type: Type[grpc.RpcError], *args):
        if self._status_code not in [None, grpc.StatusCode.OK]:
            raise error_type(
                grpc._channel._RPCState(
                    [], [], [], code=self._status_code, details=self._error_message
                ),
                *args,
            )

    def Generate(
        self, request: generation_pb2.BatchedGenerationRequest, **kwargs
    ) -> generation_pb2.BatchedGenerationResponse:
        self._maybe_raise(grpc._channel._InactiveRpcError)
        return generation_pb2.BatchedGenerationResponse()

    def GenerateStream(
        self, request: generation_pb2.SingleGenerationRequest, **kwargs
    ) -> Iterable[generation_pb2.GenerationResponse]:
        self._maybe_raise(grpc._channel._MultiThreadedRendezvous, None, None, None)
        yield generation_pb2.GenerationResponse()

    def Tokenize(
        self, request: generation_pb2.BatchedTokenizeRequest, **kwargs
    ) -> generation_pb2.BatchedTokenizeResponse:
        self._maybe_raise(grpc._channel._InactiveRpcError)
        return generation_pb2.BatchedTokenizeResponse()


## TGISGenerationClient ########################################################


@pytest.mark.parametrize(
    "status_code",
    [code for code in grpc.StatusCode if code != grpc.StatusCode.OK],
)
@pytest.mark.parametrize(
    "method", ["unary_generate", "stream_generate", "unary_tokenize"]
)
def test_TGISGenerationClient_rpc_errors(status_code, method):
    """Test that raised errors in downstream RPCs are converted to
    CaikitCoreException correctly
    """
    tgis_client = MockTgisClient(status_code)
    gen_client = tgis_utils.TGISGenerationClient(
        "foo",
        "bar",
        tgis_client,
        ProducerId("foobar"),
    )
    with pytest.raises(CaikitCoreException) as context:
        kwargs = (
            dict(
                preserve_input_text=True,
                input_tokens=True,
                generated_tokens=True,
                token_logprobs=True,
                token_ranks=True,
                max_new_tokens=20,
                min_new_tokens=20,
                truncate_input_tokens=True,
                decoding_method="GREEDY",
                top_k=None,
                top_p=None,
                typical_p=None,
                temperature=None,
                seed=None,
                repetition_penalty=0.5,
                max_time=None,
                exponential_decay_length_penalty=None,
                stop_sequences=["asdf"],
            )
            if method.endswith("_generate")
            else dict()
        )
        res = getattr(gen_client, method)(text="foobar", **kwargs)
        if method.startswith("stream_"):
            next(res)

    assert (
        context.value.status_code == tgis_utils.GRPC_TO_CAIKIT_CORE_STATUS[status_code]
    )
    rpc_err = context.value.__context__
    assert isinstance(rpc_err, grpc.RpcError)


# NOTE: This test is preserved in caikit-nlp despite being duplicated in
# caikit-tgis-backend so that we guarantee that the functionality is accessible
# in a version-compatible way here.
@pytest.mark.parametrize(
    argnames=["context", "route_info"],
    argvalues=[
        (
            fastapi.Request(
                {
                    "type": "http",
                    "headers": [
                        (tgis_utils.ROUTE_INFO_HEADER_KEY.encode(), b"sometext")
                    ],
                }
            ),
            "sometext",
        ),
        (
            fastapi.Request(
                {"type": "http", "headers": [(b"route-info", b"sometext")]}
            ),
            None,
        ),
        (
            TestServicerContext({tgis_utils.ROUTE_INFO_HEADER_KEY: "sometext"}),
            "sometext",
        ),
        (
            TestServicerContext({"route-info": "sometext"}),
            None,
        ),
        ("should raise ValueError", None),
        (None, None),
        # Uncertain how to create a grpc.ServicerContext object
    ],
)
def test_get_route_info(context: RuntimeServerContextType, route_info: Optional[str]):
    if not isinstance(context, (fastapi.Request, grpc.ServicerContext, type(None))):
        with pytest.raises(TypeError):
            tgis_utils.get_route_info(context)
    else:
        actual_route_info = tgis_utils.get_route_info(context)
        assert actual_route_info == route_info
