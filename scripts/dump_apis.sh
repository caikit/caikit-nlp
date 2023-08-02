#!/usr/bin/env bash

# Make a directory with interfaces
http_interface_dir="generated_interfaces/http"
grpc_interface_dir="generated_interfaces/grpc"
mkdir -p $http_interface_dir
mkdir -p $grpc_interface_dir

# Run the HTTP server in the background
RUNTIME_LIBRARY=caikit_nlp python -m caikit.runtime.http_server &
http_pid=$!

# Sleep for a bit and then call it to get the swagger doc
sleep 5
curl http://localhost:8080/openapi.json | jq > $http_interface_dir/openapi.json

# Kill the HTTP server and wait for it to die
kill -9 $http_pid
wait

# Dump the gRPC interfaces
RUNTIME_LIBRARY=caikit_nlp python -m caikit.runtime.dump_services $grpc_interface_dir