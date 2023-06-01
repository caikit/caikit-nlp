#!/usr/bin/env bash
# Kills the text generation launcher container if it's running
running_container_id=$(docker container ls | grep -i text-gen-server | cut -d " " -f 1)
if [ -z "$running_container_id" ]; then
      echo "TGIS container is not running; nothing to do!"
else
      echo "Trying to kill TGIS container with id: {$running_container_id}..."
      eval "docker stop $running_container_id"
fi
