#!/usr/bin/env python3
"""
gpux router — GPU multiplexer for image generation nodes.

Sits in front of N GPU nodes (each running node/serve.py).
Routes user requests to a free node: ready && !busy && empty queue.
Reuses the same node for the whole job (model stays loaded in VRAM).

Config: nodes.yaml next to this file.
"""
import asyncio, json, os, time, base64, uuid
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db as usage_db

HERE = Path(__file__).parent
NODES_FILE = HERE / 'nodes.yaml'
WEB_DIR = HERE / 'web'
HEALTH_TTL = 5.0          # seconds a health snapshot stays fresh
JOB_TIMEOUT = 900          # hard cap for one generation proxying

# central gallery service (gpux-gallery). When set, /api/gallery and
# /api/image are backed by it instead of proxying each node directly.
GALLERY_URL = os.environ.get('GPUX_GALLERY_URL', '').rstrip('/')
ROUTER_HOST = os.environ.get('GPUX_ROUTER_HOST', '127.0.0.1')
ROUTER_PORT = int(os.environ.get('GPUX_ROUTER_PORT', '8096'))
HEALTH_TIMEOUT = float(os.environ.get('GPUX_HEALTH_TIMEOUT', '4'))
NODE_TIMEOUT = float(os.environ.get('GPUX_NODE_TIMEOUT', '15'))   # job-status / media proxy to a node

def load_nodes():
    with open(NODES_FILE) as f:
        cfg = yaml.safe_load(f)
    return cfg.get('nodes', [])

NODES = load_nodes()
usage_db.init()

# ---- Horde (elastic overflow for drafts) ----
_horde_cfg = (yaml.safe_load(open(NODES_FILE)) or {}).get('horde') or {}
HORDE = None
if _horde_cfg.get('enabled'):
    from horde import HordeClient
    HORDE = HordeClient(_horde_cfg.get('api_url', 'https://aihorde.net/api/v2'),
                        _horde_cfg.get('api_key', '0000000000'),
                        _horde_cfg.get('models') or [])
_horde_jobs = {}  # jid -> {"horde_id", "status", "file", "url", "model", "started"}

def auth_user(request: Request) -> str:
    """Extract username from Basic auth header (nginx passes it)."""
    a = request.headers.get('authorization', '')
    if a.lower().startswith('basic '):
        try:
            return base64.b64decode(a.split(' ', 1)[1]).decode().split(':', 1)[0] or 'anon'
        except Exception:
            pass
    return 'anon'

app = FastAPI(title="gpux router")

# ---------------- node health ----------------
_health = {}  # name -> {"ok": bool, "ready": bool, "busy": bool, "queue": int, "ts": float, "detail": {}}

async def refresh_health():
    """Poll every node /api/status (cheap) + /api/capabilities and cache."""
    async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
        async def one(n):
            try:
                r = await client.get(n['url'] + '/api/status')
                d = r.json()
                caps = {}
                try:
                    rc = await client.get(n['url'] + '/api/capabilities')
                    if rc.status_code == 200:
                        caps = rc.json()
                except Exception:
                    pass
                return n['name'], {'ok': True, 'ready': d.get('ready', False),
                                   'busy': d.get('busy', False), 'queue': d.get('queue', 0),
                                   'detail': d, 'caps': caps, 'ts': time.time()}
            except Exception as e:
                return n['name'], {'ok': False, 'ready': False, 'busy': True,
                                   'queue': 999, 'detail': {'error': str(e)}, 'caps': {}, 'ts': time.time()}
        results = await asyncio.gather(*(one(n) for n in NODES))
    for name, snap in results:
        _health[name] = snap

async def ensure_health():
    """Use cached health; refresh synchronously only on a cold start."""
    if not _health:
        await refresh_health()

@app.on_event("startup")
async def _health_loop():
    async def loop():
        while True:
            try:
                await refresh_health()
            except Exception as e:
                print(f'[router] health loop error: {e}', flush=True)
            await asyncio.sleep(HEALTH_TTL)
    asyncio.create_task(loop())

async def pick_node(modality: str = "image", model: str | None = None, feature: str | None = None):
    """Return (node, health) of best node that can serve the request."""
    # background loop keeps _health fresh; only block on a cold start or a very stale cache
    if not _health or min((h['ts'] for h in _health.values()), default=0) < time.time() - 3 * HEALTH_TTL:
        await refresh_health()

    def can_serve(n, h):
        caps = h.get('caps') or {}
        # node without capabilities endpoint: assume image-only wangp node (backward compat)
        if not caps:
            return (modality == "image" and feature is None
                    and (model is None or model == "flux2_klein_4b"))
        if modality not in (caps.get('modalities') or []):
            return False
        if model is not None and model not in (caps.get('models') or []):
            return False
        if feature is not None and feature not in (caps.get('features') or []):
            return False
        # per-model VRAM floor (audio models declare their own footprint)
        mc = (caps.get('model_cost') or {}).get(model or '', {})
        need = mc.get('min_free_vram_gb')
        free_gb = (h.get('detail') or {}).get('free_vram_gb')
        if need is not None and free_gb is not None and free_gb < need:
            return False
        return True

    eligible = [(n, _health[n['name']]) for n in NODES
                if _health.get(n['name'], {}).get('ok') and can_serve(n, _health[n['name']])]
    if not eligible:
        want = f"model={model or 'any'} modality={modality}"
        if feature:
            want += f" feature={feature}"
        raise HTTPException(503, detail=f'no GPU nodes can serve {want}')

    def rank(t):
        h = t[1]
        caps = h.get('caps') or {}
        needs_swap = model is not None and caps.get('loaded_model') not in (None, model)
        return (needs_swap, h['busy'], h['queue'])

    free = [(n, h) for n, h in eligible
            if h['ready'] and not h['busy'] and h['queue'] == 0]
    pool = free or eligible
    pool.sort(key=rank)
    return pool[0]

def node_url(name: str) -> str:
    for n in NODES:
        if n['name'] == name:
            return n['url']
    raise HTTPException(404, detail='unknown node')

# stickiness: job id -> node name (progress/preview/job endpoints must hit same node)
_job_node = {}

# ---------------- models ----------------
class GenRequest(BaseModel):
    prompt: str
    resolution: str = "1280x960"
    steps: int = 20
    seed: int | None = None
    modality: str = "image"
    model: str | None = None
    draft: bool = False
    mode: str = "txt2img"           # "txt2img" | "img2img"
    init_image: str | None = None   # data URL / base64, forwarded verbatim to the node
    denoise: float = 0.6
    # --- audio (forwarded verbatim to the node) ---
    style: str | None = None
    negative_prompt: str | None = None
    duration_s: int | None = None
    audio_scale: float | None = None
    guidance_scale: float | None = None
    language: str | None = None
    voice_ref: str | None = None
    tts: dict | None = None

# ---------------- routes ----------------
@app.get("/api/status")
async def status():
    await ensure_health()   # cached; background loop keeps it fresh
    nodes = [{'name': n['name'], **_health.get(n['name'], {})} for n in NODES]
    free = sum(1 for x in nodes if x.get('ready') and not x.get('busy') and x.get('queue', 0) == 0)
    return {'ready': free > 0, 'nodes': nodes,
            'free': free, 'total': len(nodes)}

@app.post("/api/generate")
async def generate(req: GenRequest, request: Request):
    user = auth_user(request)
    if req.modality == "audio" and req.draft:
        raise HTTPException(400, detail='audio cannot be routed to draft/Horde overflow')
    is_img2img = req.mode == "img2img"
    if is_img2img:
        if req.draft:
            raise HTTPException(400, detail='img2img cannot be routed to draft/Horde overflow')
        if not req.init_image:
            raise HTTPException(400, detail='img2img requires init_image')
    # ---- Horde overflow for drafts ----
    if req.draft:
        if not HORDE:
            raise HTTPException(503, detail='draft overflow disabled (horde not enabled)')
        try:
            sub = await asyncio.to_thread(
                HORDE.submit, req.prompt.strip(), req.steps, req.resolution, req.seed)
        except Exception as e:
            raise HTTPException(502, detail=f'horde submit failed: {e}')
        jid = uuid.uuid4().hex[:12]
        _horde_jobs[jid] = {'horde_id': sub['horde_id'], 'status': 'queued',
                            'user': user, 'model': 'aihorde',
                            'resolution': req.resolution, 'steps': req.steps,
                            'seed': req.seed, 'started': time.time()}
        try:
            usage_db.record_job(jid, user, 'aihorde', 'aihorde', req.resolution,
                                req.steps, req.seed, time.time())
        except Exception:
            pass
        return {'id': jid, 'node': 'aihorde', 'queue': 0, 'busy': False, 'draft': True}

    node, h = await pick_node(req.modality, req.model,
                              feature='img2img' if is_img2img else None)
    body = req.model_dump()
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(node['url'] + '/api/generate', json=body)
        if r.status_code != 200:
            raise HTTPException(r.status_code, detail=r.text)
        j = r.json()
    jid = j.get('id')
    if jid:
        _job_node[jid] = node['name']
        try:
            usage_db.record_job(jid, user, node['name'],
                                req.model or 'default', req.resolution, req.steps,
                                req.seed, time.time())
        except Exception as e:
            print(f'[router] usage record failed: {e}', flush=True)
    j['node'] = node['name']
    return j

@app.get("/api/jobs/{jid}")
async def job_status(jid: str):
    # Horde jobs live locally
    if jid in _horde_jobs:
        hj = _horde_jobs[jid]
        if hj['status'] in ('queued', 'running'):
            try:
                res = await asyncio.to_thread(HORDE.check, hj['horde_id'])
                hj['status'] = res['status']
                if res['status'] == 'done':
                    hj['file'] = res['file']; hj['url'] = res['url']; hj['model'] = res['model']
                    hj['total_s'] = round(time.time() - hj['started'], 1)
                    try:
                        usage_db.finish_job(jid, 'done', hj['total_s'])
                    except Exception:
                        pass
                elif res['status'] == 'error':
                    try:
                        usage_db.finish_job(jid, 'error', None)
                    except Exception:
                        pass
            except Exception as e:
                hj['status'] = 'error'; hj['error'] = str(e)
        out = {'status': hj['status']}
        if hj['status'] == 'done':
            out.update({'files': [hj['file']], 'url': hj['url'],
                        'total': hj.get('total_s'), 'model': hj.get('model')})
        if hj.get('error'):
            out['error'] = hj['error']
        return out

    name = _job_node.get(jid)
    if not name:
        raise HTTPException(404, detail='unknown job')
    try:
        async with httpx.AsyncClient(timeout=NODE_TIMEOUT) as client:
            r = await client.get(node_url(name) + f'/api/jobs/{jid}')
        j = r.json()
    except Exception as e:
        # transient node/mesh hiccup — let the client keep polling
        print(f'[router] job_status proxy failed for {jid}: {e}', flush=True)
        return JSONResponse({'status': 'running', 'transient_error': str(e)}, status_code=200)
    if j.get('status') in ('done', 'error'):
        try:
            usage_db.finish_job(jid, j['status'], j.get('total'))
        except Exception:
            pass
    return JSONResponse(j, status_code=r.status_code)

@app.get("/api/usage")
async def usage(limit: int = 50):
    return {'jobs': usage_db.recent(limit)}

@app.get("/api/usage/summary")
async def usage_summary():
    return {'summary': usage_db.summary(), 'total': usage_db.count_all()}

@app.get("/api/events")
async def events(request: Request):
    """SSE: aggregate events from the node that owns the current job.
    Prototype: browser connects right after /api/generate, so we route to
    the node of the latest job (or first free node if none yet)."""
    latest_node = None
    if _job_node:
        latest_jid = next(reversed(_job_node))      # last submitted job id
        latest_node = _job_node.get(latest_jid)     # -> its node name
    name = latest_node or (await pick_node())[0]['name']
    url = node_url(name) + '/api/events'

    async def stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream('GET', url) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache',
                                      'X-Accel-Buffering': 'no'})

@app.get("/api/gallery")
async def gallery(limit: int = 60):
    """Merged gallery: local horde artifacts + central gallery service
    (or, in legacy mode, a fan-out to every node)."""
    out = []
    # horde artifacts (local)
    from horde import ARTIFACTS
    if ARTIFACTS.exists():
        for f in sorted(ARTIFACTS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.suffix.lower() in ('.webp', '.png', '.jpg', '.jpeg'):
                out.append({'name': f.name, 'url': f'/api/image/{f.name}',
                            'size': f.stat().st_size, 'mtime': f.stat().st_mtime,
                            'node': 'aihorde'})
    if GALLERY_URL:
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(f'{GALLERY_URL}/api/gallery', params={'limit': limit})
                out.extend(r.json().get('files', []))
        except Exception as e:
            print(f'[router] gallery service unreachable: {e}', flush=True)
    else:
        async with httpx.AsyncClient(timeout=6) as client:
            async def one(n):
                try:
                    r = await client.get(n['url'] + '/api/gallery')
                    files = r.json().get('files', [])
                    for f in files:
                        f['node'] = n['name']
                    return files
                except Exception:
                    return []
            lists = await asyncio.gather(*(one(n) for n in NODES))
        for l in lists:
            out.extend(l)
    out.sort(key=lambda f: f.get('mtime', 0), reverse=True)
    return {'files': out[:limit]}

@app.get("/api/image/{name}")
@app.get("/api/previews/{name}")
async def media(name: str, request: Request):
    # horde artifacts live locally
    if name.startswith('horde_'):
        from horde import ARTIFACTS
        f = ARTIFACTS / name
        if f.exists():
            mt = 'image/webp' if f.suffix == '.webp' else 'image/png'
            return Response(content=f.read_bytes(), media_type=mt)
        raise HTTPException(404)
    path = request.url.path  # /api/image/... or /api/previews/...
    prefer = request.query_params.get('node')
    # images -> central gallery service; previews stay per-node (transient)
    if GALLERY_URL and path.startswith('/api/image/'):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f'{GALLERY_URL}{path}',
                                     params={'node': prefer} if prefer else None)
                if r.status_code == 200:
                    return Response(content=r.content,
                                    media_type=r.headers.get('content-type', 'image/jpeg'))
        except Exception:
            pass
        raise HTTPException(404)
    # previews (and legacy images): try nodes in order
    order = [prefer] if prefer else []
    order += [n['name'] for n in NODES if n['name'] != prefer]
    async with httpx.AsyncClient(timeout=NODE_TIMEOUT) as client:
        for n in order:
            try:
                r = await client.get(node_url(n) + path)
                if r.status_code == 200:
                    return Response(content=r.content, media_type=r.headers.get('content-type', 'image/jpeg'))
            except Exception:
                continue
    raise HTTPException(404)

@app.get("/api/nodes")
async def nodes_list():
    await refresh_health()
    return {'nodes': [{'name': n['name'], 'url': n['url'], **_health.get(n['name'], {})} for n in NODES]}

@app.post("/api/nodes/register")
async def nodes_register(request: Request):
    """Onboarding: a new GPU machine registers itself (name + url).
    Persisted to nodes.yaml and used immediately."""
    try:
        body = await request.json()
        name = str(body.get('name', '')).strip()
        url = str(body.get('url', '')).strip().rstrip('/')
        if not name or not url.startswith('http'):
            raise ValueError('need {name, url}')
        if any(n['name'] == name for n in NODES):
            return {'ok': True, 'message': f'node {name} already registered'}
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    NODES.append({'name': name, 'url': url})
    # persist
    import yaml as _yaml
    _yaml.safe_dump({'nodes': NODES}, open(NODES_FILE, 'w'), sort_keys=False)
    await refresh_health()
    h = _health.get(name, {})
    return {'ok': True, 'message': f'node {name} registered',
            'reachable': h.get('ok', False),
            'ready': h.get('ready', False),
            'detail': h.get('detail', {})}

# ---------------- frontend ----------------
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=ROUTER_HOST, port=ROUTER_PORT, log_level='warning')