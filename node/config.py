#!/usr/bin/env python3
"""gpux node config — paths and model defaults (edit per machine)."""
from pathlib import Path

# WanGP installation root (contains wgp.py, shared/, ckpts/)
WAN2GP_ROOT = Path('/mnt/hdd/data/Wan2GP')

# venv with torch + wan2gp deps (must be able to import shared.api)
VENV_ACTIVATE = '/mnt/hdd/data/wan2gp_env/bin/activate'

# where the web UI binds (node side, reachable by the router)
LISTEN_HOST = '0.0.0.0'
LISTEN_PORT = 8095

# UI static files (shipped with gpux repo; repo layout: ../web, deployed: ./static)
import os as _os
_REPO_WEB = Path(__file__).resolve().parent.parent / 'web'
_LOCAL_STATIC = Path(__file__).resolve().parent / 'static'
WEB_DIR = _REPO_WEB if _REPO_WEB.exists() else _LOCAL_STATIC

# model preset used for generation
MODEL_TYPE = 'flux2_klein_4b'

# extra CLI args for WanGP init()
WAN2GP_ARGS = ["--attention", "sdpa", "--profile", "4"]

# tip: put hot checkpoints (transformer, text encoder, vae) on the fastest
# local disk and symlink them into WAN2GP_ROOT/ckpts — init drops from ~47s to ~32s

# what this node can serve (router routes by these)
CAPABILITIES = {
    "backends": ["wangp"],
    "models": [MODEL_TYPE],
    "modalities": ["image"],
    "vram_gb": 16,
    "max_resolution": "1536x1152",
}
