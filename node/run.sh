#!/bin/bash
# WanGP Studio backend launcher
export NO_PROXY="*"
export no_proxy="*"
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY ALL_PROXY
source /mnt/hdd/data/wan2gp_env/bin/activate
exec python /mnt/hdd/data/webui/serve.py "$@"