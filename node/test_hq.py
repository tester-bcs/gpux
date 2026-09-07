#!/usr/bin/env python3
"""WanGP HQ test: ginger cat in forest, high quality, with timing."""
import os, sys, time

os.environ.pop('http_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('ALL_PROXY', None)
os.environ['NO_PROXY'] = '*'

WAN2GP_ROOT = '/mnt/hdd/data/Wan2GP'
sys.path.insert(0, WAN2GP_ROOT)
os.chdir(WAN2GP_ROOT)

from pathlib import Path
from shared.api import init

t0 = time.time()
print("=== Initializing WanGP session (model load) ===", flush=True)
session = init(
    root=Path(WAN2GP_ROOT),
    cli_args=["--attention", "sdpa", "--profile", "4"],
)
t_init = time.time() - t0
print(f"=== Init + model load: {t_init:.1f}s ===", flush=True)

print("\n=== HQ Generation ===", flush=True)
print("Model: Flux 2 Klein 4B", flush=True)
print("Prompt: a ginger cat sitting in a sunlit forest, detailed fur, green eyes", flush=True)
print("Resolution: 1280x960 (HQ)", flush=True)
print("Steps: 20 (HQ)", flush=True)

settings = {
    "model_type": "flux2_klein_4b",
    "prompt": "a ginger cat sitting in a sunlit forest, detailed fur, green eyes",
    "resolution": "1280x960",
    "num_inference_steps": 20,
    "batch_size": 1,
    "embedded_guidance_scale": 1,
}

t1 = time.time()
job = session.submit_task(settings)

phases = {}
last_phase = None
phase_start = t1
last_pct = -1

for event in job.events.iter(timeout=0.3):
    if event.kind == "progress":
        p = event.data
        pct = int(p.progress * 100) if p.progress else 0
        phase = p.phase
        if phase != last_phase:
            if last_phase is not None:
                phases[last_phase] = time.time() - phase_start
            last_phase = phase
            phase_start = time.time()
            print(f"  Phase: {phase}", flush=True)
        if pct != last_pct and pct % 10 == 0:
            last_pct = pct
            print(f"    {phase}: {pct}% (step {p.current_step}/{p.total_steps})", flush=True)
    elif event.kind == "stream":
        txt = event.data.text.strip()
        if txt and 'error' in txt.lower():
            print(f"  [stderr] {txt[:200]}", flush=True)

if last_phase is not None:
    phases[last_phase] = time.time() - phase_start

result = job.result()
t_total = time.time() - t1

print("\n" + "=" * 50, flush=True)
print("TIMING REPORT", flush=True)
print("=" * 50, flush=True)
print(f"Init + model load:   {t_init:8.1f}s", flush=True)
for phase, dur in phases.items():
    print(f"  {phase:20s} {dur:8.1f}s", flush=True)
print(f"Generation total:    {t_total:8.1f}s", flush=True)
print(f"Overall:             {time.time() - t0:8.1f}s", flush=True)
print("=" * 50, flush=True)

if result.success:
    print("\nSUCCESS!", flush=True)
    for f in result.generated_files:
        print(f"   {f}", flush=True)
        if os.path.exists(f):
            print(f"   Size: {os.path.getsize(f) // 1024} KB", flush=True)
else:
    print("\nFAILED:", flush=True)
    for err in result.errors:
        print(f"   {err.message}", flush=True)