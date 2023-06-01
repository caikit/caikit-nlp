#!/usr/bin/env bash

################################################################################
# This script is the entrypoint for the multi-process runtime container that
# runs the REST gateway alongside the grpc runtime.
# The multiprocess management is intended to be handled by `tini`, the tiny
# but valid `init`.
################################################################################

set -e

echo '[STARTING RUNTIME]'
cd /app && python3 -m caikit.runtime.grpc_server &

RUNTIME_PORT=${RUNTIME_PORT:-8085}

# If TLS enabled, make an https call, otherwise make an http call
protocol="http"
if [ "${RUNTIME_TLS_SERVER_KEY}" != "" ] && [ "${RUNTIME_TLS_SERVER_CERT}" != "" ]
then
    protocol="--cacert $RUNTIME_TLS_SERVER_CERT https"
    if [ "${RUNTIME_TLS_CLIENT_CERT}" != "" ]
    then
        protocol="-k --cert $RUNTIME_TLS_SERVER_CERT --key $RUNTIME_TLS_SERVER_KEY https"
    fi
fi

# Wait for the Runtime to come up before starting the gateway
sleep 3
until $(curl --output /dev/null --silent --fail ${protocol}://localhost:${RUNTIME_PORT}); do
    echo '.'
    sleep 1
done

echo '[STARTING GATEWAY]'
PROXY_ENDPOINT="localhost:${RUNTIME_PORT}" SERVE_PORT=${GATEWAY_PORT:-8080} /gateway --swagger_path=/swagger &

wait -n
