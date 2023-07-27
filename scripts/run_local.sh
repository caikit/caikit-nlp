#!/usr/bin/env bash
cd $(dirname ${BASH_SOURCE[0]})/..
mkdir -p models

server=${SERVER:-"http"}

CONFIG_FILES=runtime_config.yaml \
LOG_LEVEL=${LOG_LEVEL:-debug3} \
LOG_FORMATTER=pretty \
python -m caikit.runtime.${server}_server
