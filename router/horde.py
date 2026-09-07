#!/usr/bin/env python3
"""gpux router — AI Horde adapter (elastic overflow for draft jobs).

Async flow: submit -> poll check -> fetch status (base64) -> save artifact.
Docs: https://aihorde.net/api/
"""
import base64
import time
from pathlib import Path

import httpx

ARTIFACTS = Path(__file__).resolve().parent / 'artifacts'
ARTIFACTS.mkdir(parents=True, exist_ok=True)

class HordeError(Exception):
    pass

class HordeClient:
    def __init__(self, api_url: str, api_key: str = '0000000000',
                 models: list[str] | None = None, timeout: int = 120):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.models = models or []
        self.timeout = timeout
        self.headers = {
            'apikey': self.api_key,
            'Client-Agent': 'gpux:0.1:tester-bcs',
            'Content-Type': 'application/json',
        }

    def submit(self, prompt: str, steps: int = 20, resolution: str = '512x512',
               seed: int | None = None) -> dict:
        try:
            w, h = resolution.lower().split('x')
            w, h = min(int(w), 1024), min(int(h), 1024)
        except Exception:
            w, h = 512, 512
        steps = max(1, min(steps, 30))
        payload = {
            'prompt': prompt,
            'params': {
                'sampler_name': 'k_euler_a',
                'cfg_scale': 7,
                'denoising_strength': 0.75,
                'height': h,
                'width': w,
                'post_processing': [],
                'steps': steps,
                'n': 1,
                'karras': True,
            },
            'nsfw': False,
            'censor_nsfw': True,
            'models': self.models,
            'r2': False,
            'shared': False,
        }
        if seed is not None:
            payload['params']['seed'] = str(seed)
        r = httpx.post(f'{self.api_url}/generate/async', json=payload,
                       headers=self.headers, timeout=self.timeout)
        if r.status_code == 401:
            raise HordeError('invalid horde api key')
        if r.status_code in (429, 503):
            raise HordeError(f'horde busy: {r.status_code}')
        r.raise_for_status()
        d = r.json()
        return {'horde_id': d['id'], 'kudos': d.get('kudos')}

    def check(self, horde_id: str) -> dict:
        r = httpx.get(f'{self.api_url}/generate/check/{horde_id}',
                      headers=self.headers, timeout=30)
        r.raise_for_status()
        d = r.json()
        if d.get('faulted'):
            raise HordeError('generation faulted on horde')
        if d.get('is_possible') is False:
            raise HordeError('request impossible for current horde workers')
        if d.get('done'):
            st = self.status(horde_id)
            gens = st.get('generations') or []
            if not gens:
                raise HordeError('no generations in status')
            gen = gens[0]
            img_b64 = gen.get('img') or ''
            if img_b64.startswith('http'):
                img_b64 = httpx.get(img_b64, timeout=60).content  # r2 url
                data = img_b64
                ext = 'webp'
            else:
                data = base64.b64decode(img_b64)
                ext = 'webp' if data[:4] == b'RIFF' else 'png'
            fname = f"horde_{gen.get('id', horde_id)}.{ext}"
            fpath = ARTIFACTS / fname
            fpath.write_bytes(data)
            return {'status': 'done', 'file': str(fpath), 'url': f'/api/image/{fname}',
                    'model': gen.get('model', 'horde'), 'worker': gen.get('worker_name')}
        if d.get('processing'):
            return {'status': 'running', 'wait_time': d.get('wait_time')}
        return {'status': 'queued', 'queue_position': d.get('queue_position'),
                'wait_time': d.get('wait_time')}

    def status(self, horde_id: str) -> dict:
        r = httpx.get(f'{self.api_url}/generate/status/{horde_id}',
                      headers=self.headers, timeout=60)
        r.raise_for_status()
        return r.json()