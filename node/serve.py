#!/usr/bin/env python3
"""
gpux node — image generation worker for one GPU machine.
Wraps WanGP API: keeps the session loaded, queues jobs, streams SSE progress.

Reads paths/settings from config.py next to this file.
"""
import os, sys, time, json, queue, threading, uuid, io, base64
from pathlib import Path

import httpx

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
from PIL import Image

OUTPUTS = Path(WAN2GP_ROOT) / 'outputs'
PREVIEWS = HERE.parent / 'previews'
PREVIEWS.mkdir(parents=True, exist_ok=True)
INPUTS = HERE.parent / 'inputs'          # user-supplied source images for img2img
INPUTS.mkdir(parents=True, exist_ok=True)
GALLERY_PENDING = HERE.parent / 'gallery_pending'   # push retries when central store is down
GALLERY_PENDING.mkdir(parents=True, exist_ok=True)
ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.webp'}
MAX_INPUT_BYTES = 20 * 1024 * 1024       # cap decoded upload size
INPUT_MAX_SIDE = 2048                    # downscale huge uploads before handing to WanGP
# idle-farming flag: when present, node reports not-ready (Horde worker holds VRAM)
FARMING_FLAG = HERE / '.farming'

# sweep stale img2img source images left over from crashed jobs (>2h old)
for _f in INPUTS.glob('*'):
    try:
        if _f.is_file() and time.time() - _f.stat().st_mtime > 7200:
            _f.unlink()
    except Exception:
        pass

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

# ---------------- central gallery push ----------------
GALLERY_URL = (getattr(config, 'GALLERY_URL', '') or '').rstrip('/')
GALLERY_TOKEN = getattr(config, 'GALLERY_TOKEN', '')
NODE_NAME = config.CAPABILITIES.get('hostname', 'node')

def _push_one(path: Path, meta: dict) -> bool:
    """POST one image + meta to the gallery service. True on success."""
    mt = 'image/jpeg' if path.suffix.lower() in ('.jpg', '.jpeg') else \
         'image/png' if path.suffix.lower() == '.png' else 'image/webp'
    with httpx.Client(timeout=30) as c:
        r = c.post(GALLERY_URL + '/ingest',
                   headers={'X-Gpux-Token': GALLERY_TOKEN},
                   data={'node': NODE_NAME, 'meta': json.dumps(meta, ensure_ascii=False)},
                   files={'file': (path.name, path.read_bytes(), mt)})
    if r.status_code == 200:
        return True
    print(f"[node] gallery push {path.name} -> {r.status_code} {r.text[:120]}", flush=True)
    return False

def _queue_pending(path: Path, meta: dict):
    try:
        (GALLERY_PENDING / (path.name + '.json')).write_text(
            json.dumps({'file': str(path), 'meta': meta}, ensure_ascii=False))
    except Exception as e:
        print(f"[node] pending-queue write failed: {e}", flush=True)

def push_to_gallery(path: str, meta: dict):
    if not GALLERY_URL:
        return
    p = Path(path)
    if not p.exists():
        return
    try:
        if _push_one(p, meta):
            print(f"[node] gallery <- {p.name}", flush=True)
        else:
            _queue_pending(p, meta)
    except Exception as e:
        print(f"[node] gallery push failed ({p.name}): {e}", flush=True)
        _queue_pending(p, meta)

def flush_pending():
    if not GALLERY_URL:
        return
    for j in list(GALLERY_PENDING.glob('*.json')):
        try:
            d = json.loads(j.read_text())
            p = Path(d['file'])
            if not p.exists():
                j.unlink(); continue
            if _push_one(p, d.get('meta', {})):
                j.unlink()
                print(f"[node] gallery <- {p.name} (retry)", flush=True)
        except Exception as e:
            print(f"[node] pending flush error for {j.name}: {e}", flush=True)

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
    flush_pending()
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
                # sidecar with the initiating prompt/params, one per output file
                for f in files:
                    try:
                        fp = Path(f)
                        (fp.parent / (fp.name + '.json')).write_text(
                            json.dumps({**job.get('meta', {}), 'ts': time.time(),
                                        'file': fp.name}, ensure_ascii=False))
                    except Exception as e:
                        print(f"[node] sidecar write failed: {e}", flush=True)
                jobs_done[jid] = {'status': 'done', 'files': files, 'timings': timings, 'total': t_total}
                broadcast({'type': 'job_done', 'id': jid, 'files': files,
                           'timings': timings, 'total': t_total})
                print(f"[node] job {jid} done in {t_total}s", flush=True)
                for f in files:
                    push_to_gallery(f, {**job.get('meta', {}), 'file': Path(f).name})
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
            src = job.get('input_file')
            if src:
                try:
                    Path(src).unlink(missing_ok=True)
                except Exception:
                    pass
            flush_pending()

threading.Thread(target=wan2gp_init_worker, daemon=True).start()
threading.Thread(target=wan2gp_job_worker, daemon=True).start()

class GenRequest(BaseModel):
    prompt: str
    resolution: str = "1280x960"
    steps: int = 20
    seed: int | None = None
    mode: str = "txt2img"           # "txt2img" | "img2img"
    init_image: str | None = None   # data URL or bare base64, required for img2img
    denoise: float = 0.6            # reserved for latent img2img (image_guide path)


def _decode_init_image(data: str, jid: str) -> Path:
    """Decode a base64 / data-URL upload to a normalized PNG under INPUTS/. Returns the path."""
    raw = data.split(',', 1)[-1] if data.startswith('data:') else data
    try:
        blob = base64.b64decode(raw, validate=True)
    except Exception:
        raise HTTPException(400, "init_image is not valid base64")
    if len(blob) > MAX_INPUT_BYTES:
        raise HTTPException(413, f"init_image too large (>{MAX_INPUT_BYTES // (1024*1024)} MB)")
    try:
        img = Image.open(io.BytesIO(blob))
        img.load()
        img = img.convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"init_image is not a readable image: {e}")
    if max(img.size) > INPUT_MAX_SIDE:
        img.thumbnail((INPUT_MAX_SIDE, INPUT_MAX_SIDE), Image.LANCZOS)
    path = INPUTS / f"{jid}.png"
    img.save(path, "PNG")
    return path.resolve()


@app.post("/api/generate")
async def api_generate(req: GenRequest):
    if not state['model_ready']:
        raise HTTPException(503, f"model loading, {round(time.time()-state['init_started'])}s elapsed")
    if not req.prompt.strip():
        raise HTTPException(400, "empty prompt")
    jid = uuid.uuid4().hex[:12]
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

    input_file = None
    if req.mode == "img2img":
        if not req.init_image:
            raise HTTPException(400, "img2img requires init_image")
        input_file = _decode_init_image(req.init_image, jid)
        # Flux 2 Klein: reference/edit path. WanGP auto-sets video_prompt_type="KI"
        # from the model's image_ref_choices when image_refs is present.
        settings["image_refs"] = [str(input_file)]

    meta = {
        "prompt": req.prompt.strip(),
        "seed": req.seed,
        "steps": settings["num_inference_steps"],
        "resolution": req.resolution,
        "mode": req.mode,
        "model": config.MODEL_TYPE,
    }
    job_queue.put({'id': jid, 'settings': settings,
                   'input_file': str(input_file) if input_file else None, 'meta': meta})
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
    offline_flag = (HERE / '.offline').exists()
    free_gb, total_gb = free_vram_gb()
    vram_ok = free_gb is None or free_gb >= config.MIN_FREE_VRAM_GB
    return {
        "ready": state['model_ready'] and not farming and vram_ok and not offline_flag,
        "farming": farming,
        "offline": offline_flag,
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
                entry = {"name": f.name, "url": f"/api/image/{f.name}",
                         "size": f.stat().st_size, "mtime": f.stat().st_mtime}
                sc = OUTPUTS / (f.name + '.json')
                if sc.exists():
                    try:
                        m = json.loads(sc.read_text())
                        for k in ('prompt', 'seed', 'steps', 'mode'):
                            if m.get(k) is not None:
                                entry[k] = m[k]
                    except Exception:
                        pass
                files.append(entry)
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