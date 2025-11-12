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
from os import getenv, path
import json
import sys

# Third Party
from config import host, port
from google.protobuf.struct_pb2 import Struct
import grpc
import requests

# First Party
from caikit.config.config import get_config
from caikit.runtime.service_factory import ServicePackageFactory
import caikit

if __name__ == "__main__":
    model_id = getenv("MODEL", "mini")

    # Add the runtime/library to the path
    sys.path.append(path.abspath(path.join(path.dirname(__file__), "../../")))

    # Load configuration for Caikit runtime
    CONFIG_PATH = path.realpath(path.join(path.dirname(__file__), "config.yml"))
    caikit.configure(CONFIG_PATH)

    inference_service = ServicePackageFactory().get_service_package(
        ServicePackageFactory.ServiceType.INFERENCE,
    )

    top_n = 3
    queries = ["first sentence", "any sentence"]
    documents = [
        {"text": "first sentence", "title": "first title"},
        {"_text": "another sentence", "more": "more attributes here"},
        {
            "text": "a doc with a nested metadata",
            "meta": {"foo": "bar", "i": 999, "f": 12.34},
        },
    ]

    print("======================")
    print("TOP N: ", top_n)
    print("QUERIES: ", queries)
    print("DOCUMENTS: ", documents)
    print("======================")

    if get_config().runtime.grpc.enabled:

        # Setup the client
        channel = grpc.insecure_channel(f"{host}:{port}")
        client_stub = inference_service.stub_class(channel)

        # gRPC JSON documents go in Structs
        docs = []
        for d in documents:
            s = Struct()
            s.update(d)
            docs.append(s)

        request = inference_service.messages.RerankTasksRequest(
            queries=queries, documents=docs, top_n=top_n
        )
        response = client_stub.RerankTasksPredict(
            request, metadata=[("mm-model-id", model_id)], timeout=1
        )

        # print("RESPONSE:", response)

        # gRPC response
        print("RESPONSE from gRPC:")
        for i, r in enumerate(response.results):
            print("===")
            print("QUERY: ", r.query)
            for s in r.scores:
                print(f"  score: {s.score}  index: {s.index}  text: {s.text}")

    if get_config().runtime.http.enabled:
        # REST payload
        payload = {
            "inputs": {
                "documents": documents,
                "queries": queries,
            },
            "parameters": {
                "top_n": -1,
                "return_documents": True,
                "return_queries": True,
                "return_text": True,
            },
            "model_id": model_id,
        }
        response = requests.post(
            f"http://{host}:8080/api/v1/task/rerank-tasks",
            json=payload,
            timeout=1,
        )
        print("===================")
        print("RESPONSE from HTTP:")
        print(json.dumps(response.json(), indent=4))
