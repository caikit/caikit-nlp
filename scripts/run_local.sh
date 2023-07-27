#!/usr/bin/env bash
mkdir -p models

cd $(dirname ${BASH_SOURCE[0]})/..

server=${SERVER:-"http"}

CONFIG_FILES=runtime_config.yaml \
LOG_LEVEL=${LOG_LEVEL:-debug3} \
LOG_FORMATTER=pretty \
python -m caikit.runtime.${server}_server
