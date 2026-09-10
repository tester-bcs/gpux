#!/usr/bin/env python3
"""
gpux gallery — central image store for all GPU nodes.

Nodes POST finished images + metadata to /ingest; the router (and, later,
the frontend) read them back via /api/gallery and /api/image/<name>.

Storage layout:  <STORE>/<node>/<filename>         the image
                 <STORE>/<node>/<filename>.json    sidecar {prompt, seed, steps, mode, ...}

Config via env:
  GPUX_STORE          store root (default: ./store next to this file)
  GPUX_INGEST_TOKEN   shared secret; nodes must send it as X-Gpux-Token
  GPUX_GALLERY_HOST   bind host  (default 127.0.0.1)
  GPUX_GALLERY_PORT   bind port  (default 8097)
"""
import json, os, time
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
from fastapi.responses import Response, JSONResponse, FileResponse

HERE = Path(__file__).parent
STORE = Path(os.environ.get("GPUX_STORE", HERE / "store"))
STORE.mkdir(parents=True, exist_ok=True)
TOKEN = os.environ.get("GPUX_INGEST_TOKEN", "")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg"}
MEDIA_EXT = IMG_EXT | AUDIO_EXT
MAX_BYTES = 80 * 1024 * 1024
META_KEYS = ("prompt", "seed", "steps", "mode", "resolution", "model",
             "modality", "kind", "duration_s")

app = FastAPI(title="gpux gallery")


def _safe_seg(s: str) -> str:
    """One path segment, no traversal, no separators."""
    s = os.path.basename((s or "").strip()).strip()
    if not s or s in (".", "..") or "/" in s or "\\" in s:
        raise HTTPException(400, "bad name")
    return s


def _mt(name: str) -> str:
    e = Path(name).suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".webp": "image/webp", ".wav": "audio/wav", ".mp3": "audio/mpeg",
            ".flac": "audio/flac", ".ogg": "audio/ogg"}.get(e, "application/octet-stream")


@app.get("/healthz")
async def healthz():
    nodes = sorted(p.name for p in STORE.iterdir() if p.is_dir()) if STORE.exists() else []
    return {"ok": True, "store": str(STORE), "nodes": nodes}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...),
                 node: str = Form(...),
                 meta: str = Form("{}"),
                 x_gpux_token: str = Header(default="")):
    if TOKEN and x_gpux_token != TOKEN:
        raise HTTPException(401, "bad token")
    node = _safe_seg(node)
    name = _safe_seg(file.filename or "")
    if Path(name).suffix.lower() not in MEDIA_EXT:
        raise HTTPException(400, "unsupported extension")

    blob = await file.read()
    if not blob:
        raise HTTPException(400, "empty file")
    if len(blob) > MAX_BYTES:
        raise HTTPException(413, "file too large")

    try:
        m = json.loads(meta) if meta else {}
        if not isinstance(m, dict):
            m = {}
    except Exception:
        m = {}
    m = {k: m[k] for k in META_KEYS if k in m and m[k] is not None}
    m["node"] = node
    m["file"] = name
    m["ingested_ts"] = time.time()

    d = STORE / node
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (name + ".part")
    tmp.write_bytes(blob)
    tmp.rename(d / name)
    (d / (name + ".json")).write_text(json.dumps(m, ensure_ascii=False))
    return {"ok": True, "node": node, "name": name, "bytes": len(blob)}


def _iter_images():
    if not STORE.exists():
        return
    for nd in STORE.iterdir():
        if not nd.is_dir():
            continue
        for f in nd.iterdir():
            if f.suffix.lower() in MEDIA_EXT and f.is_file():
                yield nd.name, f


@app.get("/api/gallery")
async def gallery(limit: int = 60):
    rows = []
    for node, f in _iter_images():
        try:
            st = f.stat()
        except OSError:
            continue
        entry = {"name": f.name, "url": f"/api/image/{f.name}",
                 "size": st.st_size, "mtime": st.st_mtime, "node": node}
        sc = f.with_name(f.name + ".json")
        if sc.exists():
            try:
                m = json.loads(sc.read_text())
                for k in ("prompt", "seed", "steps", "mode", "modality", "kind", "duration_s"):
                    if m.get(k) is not None:
                        entry[k] = m[k]
            except Exception:
                pass
        rows.append(entry)
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return {"files": rows[:max(1, limit)]}


@app.get("/api/image/{name}")
async def image(name: str, node: str | None = None):
    name = _safe_seg(name)
    if Path(name).suffix.lower() not in MEDIA_EXT:
        raise HTTPException(404)
    if node:
        cand = [STORE / _safe_seg(node) / name]
    else:
        cand = [STORE / nd.name / name for nd in STORE.iterdir() if nd.is_dir()]
    for p in cand:
        if p.is_file():
            # FileResponse supports HTTP Range -> <audio> seeking
            return FileResponse(p, media_type=_mt(name),
                                headers={"Cache-Control": "public, max-age=86400"})
    raise HTTPException(404)


@app.get("/api/meta/{name}")
async def meta_one(name: str, node: str | None = None):
    name = _safe_seg(name)
    dirs = [STORE / _safe_seg(node)] if node else [d for d in STORE.iterdir() if d.is_dir()]
    for d in dirs:
        sc = d / (name + ".json")
        if sc.is_file():
            try:
                return JSONResponse(json.loads(sc.read_text()))
            except Exception:
                break
    raise HTTPException(404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,
                host=os.environ.get("GPUX_GALLERY_HOST", "127.0.0.1"),
                port=int(os.environ.get("GPUX_GALLERY_PORT", "8097")),
                log_level="warning")
