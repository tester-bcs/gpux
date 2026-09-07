#!/usr/bin/env python3
"""
gpux node — image generation worker for one GPU machine.
Wraps WanGP API: keeps the session loaded, queues jobs, streams SSE progress.

Reads paths/settings from config.py next to this file.
"""
import os, sys, time, json, queue, threading, uuid
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config

# Kill SOCKS proxy for httpx inside WanGP/gradio stack
for k in ('http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY', 'ALL_PROXY'):
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'

WAN2GP_ROOT = str(config.WAN2GP_ROOT)
sys.path.insert(0, WAN2GP_ROOT)
os.chdir(WAN2GP_ROOT)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

OUTPUTS = Path(WAN2GP_ROOT) / 'outputs'
PREVIEWS = HERE.parent / 'previews'
PREVIEWS.mkdir(parents=True, exist_ok=True)
ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.webp'}
# idle-farming flag: when present, node reports not-ready (Horde worker holds VRAM)
FARMING_FLAG = HERE / '.farming'

app = FastAPI(title="gpux node")

state = {
    'session': None, 'model_ready': False, 'init_started': time.time(),
    'init_error': None, 'busy': False, 'queue_len': 0, 'current_job': None,
}
job_queue = queue.Queue()
listeners = []
jobs_done = {}

def broadcast(event: dict):
    dead = []
    for q in listeners:
        try:
            q.put_nowait(event)
        except Exception:
            dead.append(q)
    for q in dead:
        listeners.remove(q)

def wan2gp_init_worker():
    try:
        from shared.api import init
        session = init(root=Path(WAN2GP_ROOT), cli_args=config.WAN2GP_ARGS)
        state['session'] = session
        state['model_ready'] = True
        broadcast({'type': 'ready', 'elapsed': round(time.time() - state['init_started'], 1)})
        print(f"[node] WanGP ready in {time.time()-state['init_started']:.1f}s", flush=True)
    except Exception as e:
        state['init_error'] = str(e)
        broadcast({'type': 'init_error', 'error': str(e)})
        print(f"[node] init FAILED: {e}", flush=True)

def wan2gp_job_worker():
    while True:
        job = job_queue.get()
        jid = job['id']
        state.update(busy=True, current_job=jid, queue_len=job_queue.qsize())
        broadcast({'type': 'job_start', 'id': jid, 'queue_len': state['queue_len']})

        timings, last_phase, phase_start = {}, None, time.time()
        t1 = time.time()
        try:
            task = state['session'].submit_task(job['settings'])
            preview_n = 0
            for ev in task.events.iter(timeout=0.3):
                if ev.kind == 'progress':
                    p = ev.data
                    if p.phase != last_phase:
                        if last_phase:
                            timings[last_phase] = round(time.time() - phase_start, 1)
                        last_phase, phase_start = p.phase, time.time()
                    pct = min(int((p.progress or 0) * 100), 100)
                    broadcast({'type': 'progress', 'id': jid, 'phase': p.phase,
                               'step': f"{p.current_step or 0}/{p.total_steps or 0}", 'pct': pct})
                elif ev.kind == 'preview':
                    img = ev.data.image
                    if img is not None:
                        preview_n += 1
                        ppath = PREVIEWS / f"{jid}_p{preview_n}.jpg"
                        img.save(ppath, quality=70)
                        broadcast({'type': 'preview', 'id': jid,
                                   'url': f"/api/previews/{ppath.name}"})
                elif ev.kind == 'stream':
                    txt = ev.data.text.strip()
                    if txt and 'error' in txt.lower():
                        broadcast({'type': 'log', 'id': jid, 'text': txt[:200]})
            if last_phase:
                timings[last_phase] = round(time.time() - phase_start, 1)

            result = task.result()
            t_total = round(time.time() - t1, 1)
            if result.success:
                files = [str(f) for f in result.generated_files if Path(f).exists()]
                jobs_done[jid] = {'status': 'done', 'files': files, 'timings': timings, 'total': t_total}
                broadcast({'type': 'job_done', 'id': jid, 'files': files,
                           'timings': timings, 'total': t_total})
                print(f"[node] job {jid} done in {t_total}s", flush=True)
            else:
                errs = '; '.join(e.message for e in result.errors)
                jobs_done[jid] = {'status': 'error', 'error': errs, 'total': t_total}
                broadcast({'type': 'job_error', 'id': jid, 'error': errs})
                print(f"[node] job {jid} FAILED: {errs}", flush=True)
        except Exception as e:
            jobs_done[jid] = {'status': 'error', 'error': str(e)}
            broadcast({'type': 'job_error', 'id': jid, 'error': str(e)})
            print(f"[node] job {jid} exception: {e}", flush=True)
        finally:
            state.update(busy=False, current_job=None)

threading.Thread(target=wan2gp_init_worker, daemon=True).start()
threading.Thread(target=wan2gp_job_worker, daemon=True).start()

class GenRequest(BaseModel):
    prompt: str
    resolution: str = "1280x960"
    steps: int = 20
    seed: int | None = None

@app.post("/api/generate")
async def api_generate(req: GenRequest):
    if not state['model_ready']:
        raise HTTPException(503, f"model loading, {round(time.time()-state['init_started'])}s elapsed")
    if not req.prompt.strip():
        raise HTTPException(400, "empty prompt")
    settings = {
        "model_type": config.MODEL_TYPE,
        "prompt": req.prompt.strip(),
        "resolution": req.resolution,
        "num_inference_steps": max(1, min(50, req.steps)),
        "batch_size": 1,
        "embedded_guidance_scale": 1,
    }
    if req.seed is not None:
        settings["seed"] = req.seed
    jid = uuid.uuid4().hex[:12]
    job_queue.put({'id': jid, 'settings': settings})
    state['queue_len'] = job_queue.qsize()
    return {"id": jid, "queue": state['queue_len'], "busy": state['busy']}

def free_vram_gb():
    """Actual free VRAM on GPU0 (accounts for games/render/apps)."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return round(free / 1024**3, 1), round(total / 1024**3, 1)
    except Exception:
        pass
    return None, None

@app.get("/api/status")
async def api_status():
    farming = FARMING_FLAG.exists()
    free_gb, total_gb = free_vram_gb()
    vram_ok = free_gb is None or free_gb >= config.MIN_FREE_VRAM_GB
    return {
        "ready": state['model_ready'] and not farming and vram_ok,
        "farming": farming,
        "busy": state['busy'],
        "queue": state['queue_len'], "current": state['current_job'],
        "init_elapsed": round(time.time() - state['init_started'], 1),
        "init_error": state['init_error'],
        "free_vram_gb": free_gb, "total_vram_gb": total_gb,
        "vram_ok": vram_ok,
    }

@app.get("/api/capabilities")
async def api_capabilities():
    caps = dict(config.CAPABILITIES)
    free_gb, _ = free_vram_gb()
    if free_gb is not None:
        caps["free_vram_gb"] = free_gb
    return caps

@app.get("/api/events")
async def api_events(request: Request):
    q = queue.Queue()
    listeners.append(q)

    async def gen():
        try:
            yield f"data: {json.dumps({'type': 'hello', 'ready': state['model_ready'], 'busy': state['busy']})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = q.get(timeout=1)
                    yield f"data: {json.dumps(ev)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            if q in listeners:
                listeners.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.get("/api/jobs/{jid}")
async def api_job(jid: str):
    if jid in jobs_done:
        return jobs_done[jid]
    if state['current_job'] == jid:
        return {"status": "running"}
    return {"status": "queued"}

@app.get("/api/gallery")
async def api_gallery(limit: int = 60):
    files = []
    if OUTPUTS.exists():
        for f in sorted(OUTPUTS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.suffix.lower() in ALLOWED_EXT:
                files.append({"name": f.name, "url": f"/api/image/{f.name}",
                              "size": f.stat().st_size, "mtime": f.stat().st_mtime})
            if len(files) >= limit:
                break
    return {"files": files}

@app.get("/api/image/{name}")
async def api_image(name: str):
    f = OUTPUTS / name
    if not f.exists() or f.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(404)
    return FileResponse(f, media_type="image/jpeg")

@app.get("/api/previews/{name}")
async def api_preview(name: str):
    f = PREVIEWS / name
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="image/jpeg")

app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.LISTEN_HOST, port=config.LISTEN_PORT, log_level="warning")