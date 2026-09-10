#!/usr/bin/env python3
"""gpux node config — ms-7c75 (RTX 5060 Ti 16GB)."""
import os
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

# image model preset (loaded at boot; WanGP swaps model_type per task)
MODEL_TYPE = 'flux2_klein_4b'
DEFAULT_MODEL = MODEL_TYPE

# audio models this node can serve. Deliberately the *light* set — swapping to
# any of these from klein is RAM-safe on 31GB/full-swap. Heavy variants
# (ace_step_v1_5_turbo_lm_4b, stable_audio3_medium/FA2) wait for a RAM upgrade.
# alias -> {wangp_model_type, kind, defaults, limits}
AUDIO_MODELS = {
    "stable_audio_sfx": {
        "wangp_model_type": "stable_audio3_small_sfx", "kind": "sfx", "output_ext": "wav",
        "steps_range": (6, 16), "steps_default": 8, "max_duration_s": 120,
        "min_free_vram_gb": 4.0, "load_time_s": 25,
        "defaults": {"audio_scale": 0.9, "guidance_scale": 1.0,
                     "negative_prompt": "poor quality, distorted, noisy",
                     "sample_solver": "pingpong"},
    },
    "stable_audio": {
        "wangp_model_type": "stable_audio3_small", "kind": "music", "output_ext": "wav",
        "steps_range": (6, 16), "steps_default": 8, "max_duration_s": 120,
        "min_free_vram_gb": 4.0, "load_time_s": 25,
        "defaults": {"audio_scale": 0.9, "guidance_scale": 1.0,
                     "negative_prompt": "poor quality, distorted, noisy",
                     "sample_solver": "pingpong"},
    },
    "ace_step": {
        "wangp_model_type": "ace_step_v1_5", "kind": "music", "output_ext": "wav",
        "steps_range": (6, 16), "steps_default": 8, "max_duration_s": 240,
        "min_free_vram_gb": 7.0, "load_time_s": 45,
        "defaults": {"audio_scale": 0.5, "guidance_scale": 1.0, "shift": 1.0,
                     "scheduler_type": "euler"},
    },
    "chatterbox": {
        "wangp_model_type": "chatterbox", "kind": "tts", "output_ext": "wav",
        "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "ru", "zh", "ja"],
        "voice_ref": True, "min_free_vram_gb": 3.0, "load_time_s": 20,
        "defaults": {"audio_prompt_type": "", "model_mode": "en", "temperature": 0.8,
                     "num_inference_steps": 0, "video_length": 0,
                     "custom_settings": {"exaggeration": 0.5, "pace": 0.5}},
    },
}

# extra CLI args for WanGP init()
WAN2GP_ARGS = ["--attention", "sdpa", "--profile", "4"]

# min free VRAM (GB) for the node to accept tasks — if a game/render eats the GPU,
# the node honestly goes offline in /api/status
MIN_FREE_VRAM_GB = 6.0

# central gallery service (gpux-gallery on niceguy) — finished images are pushed here.
# Empty GALLERY_URL disables the push (node keeps only local outputs/).
GALLERY_URL = os.environ.get('GPUX_GALLERY_URL', 'http://100.64.0.2:8097')
_tok = Path(__file__).resolve().parent / 'gallery.token'
GALLERY_TOKEN = (_tok.read_text().strip() if _tok.exists()
                 else os.environ.get('GPUX_GALLERY_TOKEN', ''))

# what this node can serve (router routes by these)
CAPABILITIES = {
    "backends": ["wangp"],
    "models": [MODEL_TYPE] + list(AUDIO_MODELS),
    "modalities": ["image", "audio"],
    "features": ["img2img"],
    "vram_gb": 16,
    "max_resolution": "1536x1152",
    "hostname": "ms-7c75",
}