#!/usr/bin/env python3
"""WanGP test: generate an image of a ginger cat in a forest, 640x480."""
import os, sys, time

# Kill SOCKS proxy for httpx/gradio
os.environ.pop('http_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('ALL_PROXY', None)
os.environ['NO_PROXY'] = '*'

# Add WanGP root to path
WAN2GP_ROOT = '/mnt/hdd/data/Wan2GP'
sys.path.insert(0, WAN2GP_ROOT)
os.chdir(WAN2GP_ROOT)

from pathlib import Path
from shared.api import init

print("=== Initializing WanGP session ===", flush=True)
session = init(
    root=Path(WAN2GP_ROOT),
    cli_args=["--attention", "sdpa", "--profile", "4"],
)

print("\n=== Generating image ===", flush=True)
print("Model: Flux 2 Klein 4B", flush=True)
print("Prompt: a ginger cat sitting in a sunlit forest, detailed fur, green eyes", flush=True)
print("Resolution: 640x480", flush=True)

settings = {
    "model_type": "flux2_klein_4b",
    "prompt": "a ginger cat sitting in a sunlit forest, detailed fur, green eyes",
    "resolution": "640x480",
    "num_inference_steps": 4,
    "batch_size": 1,
    "embedded_guidance_scale": 1,
}

job = session.submit_task(settings)

last_progress = -1
for event in job.events.iter(timeout=0.3):
    if event.kind == "progress":
        p = event.data
        pct = int(p.progress * 100) if p.progress else 0
        if pct != last_progress and pct % 5 == 0:
            last_progress = pct
            print(f"  Progress: {p.phase} | step {p.current_step}/{p.total_steps} | {pct}%", flush=True)
    elif event.kind == "stream":
        line = event.data
        txt = line.text.strip()
        if txt and ('%' in txt or 'ownload' in txt or 'Loading' in txt or 'error' in txt.lower()):
            print(f"  [{line.stream}] {txt[:150]}", flush=True)

print("\n=== Waiting for result ===", flush=True)
result = job.result()

if result.success:
    print(f"\nSUCCESS! Generated files:", flush=True)
    for f in result.generated_files:
        print(f"   {f}", flush=True)
        if os.path.exists(f):
            print(f"   Size: {os.path.getsize(f) // 1024} KB", flush=True)
else:
    print(f"\nFAILED:", flush=True)
    for err in result.errors:
        print(f"   {err.message}", flush=True)

print("\nDone.", flush=True)