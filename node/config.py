#!/usr/bin/env python3
"""gpux node config — mini-pc (Илья, RTX 5070 Ti 16GB)."""
from pathlib import Path

# WanGP installation root (contains wgp.py, shared/, ckpts/)
WAN2GP_ROOT = Path('/home/avk/Wan2GP')

# venv with torch + wan2gp deps (must be able to import shared.api)
VENV_ACTIVATE = '/home/avk/Wan2GP/venv/bin/activate'

# where the web UI binds (node side, reachable by the router)
LISTEN_HOST = '0.0.0.0'
LISTEN_PORT = 8095

# UI static files (shipped with gpux repo; repo layout: ../web, deployed: ./static)
_REPO_WEB = Path(__file__).resolve().parent.parent / 'web'
_LOCAL_STATIC = Path(__file__).resolve().parent / 'static'
WEB_DIR = _REPO_WEB if _REPO_WEB.exists() else _LOCAL_STATIC

# model preset used for generation
MODEL_TYPE = 'flux2_klein_4b'

# extra CLI args for WanGP init()
WAN2GP_ARGS = ["--attention", "sdpa", "--profile", "4"]

# min free VRAM (GB) for the node to accept tasks — if a game/render eats the GPU,
# the node honestly goes offline in /api/status
MIN_FREE_VRAM_GB = 6.0

# what this node can serve (router routes by these)
CAPABILITIES = {
    "backends": ["wangp"],
    "models": [MODEL_TYPE],
    "modalities": ["image"],
    "features": ["img2img"],
    "vram_gb": 16,
    "max_resolution": "1536x1152",
    "hostname": "ms-7c75",
}