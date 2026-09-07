#!/usr/bin/env python3
"""gpux idle watcher — turns this node into an AI Horde worker when idle.

Logic loop (every 60s):
  - node busy or queue > 0  -> stop Horde worker, drain, back to warm mode
  - node idle >= IDLE_MIN   -> start Horde worker (horde-worker-reGen)
Config: /home/avk/gpux-node/idle.env (IDLE_MINUTES, HORDE_API_KEY, ...)
"""
import os, sys, time, signal, subprocess, urllib.request, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_FILE = HERE / 'idle.env'
LOG = HERE / 'idle_worker.log'

def load_env():
    cfg = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                cfg[k.strip()] = v.strip()
    return cfg

CFG = load_env()
IDLE_MIN = int(CFG.get('IDLE_MINUTES', 30))
REGEN_DIR = Path(CFG.get('HORDE_WORKER_DIR', '/mnt/hdd/data/horde-worker-reGen'))
NODE_URL = CFG.get('NODE_URL', 'http://127.0.0.1:8095')

def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def node_status():
    try:
        with urllib.request.urlopen(f'{NODE_URL}/api/status', timeout=5) as r:
            return json.loads(r.read())
    except Exception as e:
        log(f'status unreachable: {e}')
        return None

worker_proc = None

def start_worker():
    global worker_proc
    env = os.environ.copy()
    env['NO_PROXY'] = '*'
    env['HF_HOME'] = str(REGEN_DIR / 'hf_cache')
    # mark node as farming (router stops routing tasks here until flag removed)
    Path(HERE / '.farming').touch()
    log('farming flag set, starting horde-worker-reGen...')
    worker_proc = subprocess.Popen(
        ['./venv/bin/python', '-u', 'run_worker.py'],
        cwd=str(REGEN_DIR), env=env,
        stdout=open(LOG, 'a'), stderr=subprocess.STDOUT,
        start_new_session=True)
    log(f'worker pid={worker_proc.pid}')

def stop_worker():
    global worker_proc
    if worker_proc is None:
        return
    log('stopping horde worker (draining)...')
    try:
        os.killpg(os.getpgid(worker_proc.pid), signal.SIGTERM)
    except Exception:
        try: worker_proc.terminate()
        except Exception: pass
    try:
        worker_proc.wait(timeout=120)
    except Exception:
        try:
            os.killpg(os.getpgid(worker_proc.pid), signal.SIGKILL)
        except Exception:
            pass
    worker_proc = None
    # wait VRAM actually freed, then un-mark node
    time.sleep(5)
    (HERE / '.farming').unlink(missing_ok=True)
    log('worker stopped, farming flag removed, VRAM free')

def worker_alive():
    return worker_proc is not None and worker_proc.poll() is None

def main():
    # cleanup orphaned state from previous run
    if (HERE / '.farming').exists():
        log('removing stale farming flag, killing orphaned bridge.py if any')
        subprocess.run(['pkill', '-f', 'horde-worker-reGen/run_worker.py'], check=False)
        (HERE / '.farming').unlink(missing_ok=True)
    log(f'idle watcher started (idle_min={IDLE_MIN}, node={NODE_URL})')
    idle_since = time.time()
    while True:
        st = node_status()
        busy = bool(st and (st.get('busy') or st.get('queue', 0) > 0))
        ready = bool(st and st.get('ready'))

        if busy or not ready:
            idle_since = time.time()
            if worker_alive():
                stop_worker()
        else:
            idle_for = (time.time() - idle_since) / 60
            if idle_for >= IDLE_MIN and not worker_alive():
                log(f'idle for {idle_for:.0f}m -> farming')
                start_worker()

        time.sleep(60)

if __name__ == '__main__':
    main()