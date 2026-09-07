#!/usr/bin/env python3
"""
gpux router — GPU multiplexer for image generation nodes.

Sits in front of N GPU nodes (each running node/serve.py).
Routes user requests to a free node: ready && !busy && empty queue.
Reuses the same node for the whole job (model stays loaded in VRAM).

Config: nodes.yaml next to this file.
"""
import asyncio, json, os, time
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).parent
NODES_FILE = HERE / 'nodes.yaml'
WEB_DIR = HERE / 'web'
HEALTH_TTL = 5.0          # seconds a health snapshot stays fresh
JOB_TIMEOUT = 900          # hard cap for one generation proxying

def load_nodes():
    with open(NODES_FILE) as f:
        cfg = yaml.safe_load(f)
    return cfg.get('nodes', [])

NODES = load_nodes()

app = FastAPI(title="gpux router")

# ---------------- node health ----------------
_health = {}  # name -> {"ok": bool, "ready": bool, "busy": bool, "queue": int, "ts": float, "detail": {}}

async def refresh_health():
    """Poll every node /api/status (cheap) and cache."""
    async with httpx.AsyncClient(timeout=4) as client:
        async def one(n):
            try:
                r = await client.get(n['url'] + '/api/status')
                d = r.json()
                return n['name'], {'ok': True, 'ready': d.get('ready', False),
                                   'busy': d.get('busy', False), 'queue': d.get('queue', 0),
                                   'detail': d, 'ts': time.time()}
            except Exception as e:
                return n['name'], {'ok': False, 'ready': False, 'busy': True,
                                   'queue': 999, 'detail': {'error': str(e)}, 'ts': time.time()}
        results = await asyncio.gather(*(one(n) for n in NODES))
    for name, snap in results:
        _health[name] = snap

async def pick_node():
    """Return (node, health) of best free node, refreshing if stale."""
    if not _health or min((h['ts'] for h in _health.values()), default=0) < time.time() - HEALTH_TTL:
        await refresh_health()
    # prefer: ready, not busy, smallest queue; fall back to ready-but-busy only if nothing free
    free = [(n, _health[n['name']]) for n in NODES
            if _health.get(n['name'], {}).get('ok') and _health[n['name']]['ready']
            and not _health[n['name']]['busy'] and _health[n['name']]['queue'] == 0]
    if free:
        return free[0]
    ready_busy = [(n, _health[n['name']]) for n in NODES
                  if _health.get(n['name'], {}).get('ok') and _health[n['name']]['ready']]
    if ready_busy:
        ready_busy.sort(key=lambda t: (t[1]['busy'], t[1]['queue']))
        return ready_busy[0]
    raise HTTPException(503, detail='no GPU nodes available')

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

# ---------------- routes ----------------
@app.get("/api/status")
async def status():
    await refresh_health()
    nodes = [{'name': n['name'], **_health.get(n['name'], {})} for n in NODES]
    free = sum(1 for x in nodes if x.get('ready') and not x.get('busy') and x.get('queue', 0) == 0)
    return {'ready': free > 0, 'nodes': nodes,
            'free': free, 'total': len(nodes)}

@app.post("/api/generate")
async def generate(req: GenRequest):
    node, h = await pick_node()
    body = req.model_dump()
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(node['url'] + '/api/generate', json=body)
        if r.status_code != 200:
            raise HTTPException(r.status_code, detail=r.text)
        j = r.json()
    jid = j.get('id')
    if jid:
        _job_node[jid] = node['name']
    j['node'] = node['name']
    return j

@app.get("/api/jobs/{jid}")
async def job_status(jid: str):
    name = _job_node.get(jid)
    if not name:
        raise HTTPException(404, detail='unknown job')
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get(node_url(name) + f'/api/jobs/{jid}')
        return JSONResponse(r.json(), status_code=r.status_code)

@app.get("/api/events")
async def events(request: Request):
    """SSE: aggregate events from the node that owns the current job.
    Prototype: browser connects right after /api/generate, so we route to
    the node of the latest job (or first free node if none yet)."""
    latest = None
    if _job_node:
        latest = next(reversed(_job_node))  # last submitted job's node
    name = latest or (await pick_node())[0]['name']
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
async def gallery(node: str | None = None):
    """Merged gallery from all reachable nodes."""
    out = []
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
    return {'files': out[:60]}

@app.get("/api/image/{name}")
@app.get("/api/previews/{name}")
async def media(name: str, request: Request):
    # previews/images live per node; try nodes in order (gallery tells which one via ?node=)
    prefer = request.query_params.get('node')
    order = [prefer] if prefer else []
    order += [n['name'] for n in NODES if n['name'] != prefer]
    path = request.url.path  # /api/image/... or /api/previews/...
    async with httpx.AsyncClient(timeout=10) as client:
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

# ---------------- frontend ----------------
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8096, log_level='warning')