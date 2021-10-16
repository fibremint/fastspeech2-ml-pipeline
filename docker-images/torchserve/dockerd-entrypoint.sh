#!/bin/bash
# ref: https://github.com/pytorch/serve/blob/master/docker/dockerd-entrypoint.sh
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --start --model-store /home/model-server/model-store --ts-config /home/model-server/config.properties
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null