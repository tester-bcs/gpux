#!/usr/bin/env python3
"""gpux router — usage accounting (sqlite)."""
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / 'usage.db'

SCHEMA = '''CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY,
  ts REAL,
  user TEXT DEFAULT 'anon',
  node TEXT,
  model TEXT,
  resolution TEXT,
  steps INTEGER,
  seed INTEGER,
  status TEXT DEFAULT 'queued',
  total_s REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_ts ON jobs(ts);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user);
'''

def conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    return c

def init():
    with conn() as c:
        c.executescript(SCHEMA)

def record_job(jid: str, user: str, node: str, model: str,
               resolution: str, steps: int, seed, ts: float):
    with conn() as c:
        c.execute('INSERT OR REPLACE INTO jobs(id, ts, user, node, model, resolution, steps, seed, status) '
                  'VALUES(?,?,?,?,?,?,?,?,?)',
                  (jid, ts, user, node, model, resolution, steps, seed, 'queued'))

def finish_job(jid: str, status: str, total_s: float | None):
    with conn() as c:
        c.execute('UPDATE jobs SET status=?, total_s=? WHERE id=?', (status, total_s, jid))

def recent(limit: int = 50):
    with conn() as c:
        rows = c.execute('SELECT id, ts, user, node, model, resolution, steps, seed, status, total_s '
                         'FROM jobs ORDER BY ts DESC LIMIT ?', (limit,)).fetchall()
        return [dict(r) for r in rows]

def summary():
    with conn() as c:
        rows = c.execute('''
            SELECT user,
                   COUNT(*) AS jobs,
                   SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done,
                   SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors,
                   COALESCE(SUM(steps),0) AS steps,
                   ROUND(COALESCE(SUM(total_s),0),1) AS gpu_seconds
            FROM jobs GROUP BY user ORDER BY jobs DESC''').fetchall()
        return [dict(r) for r in rows]

def count_all() -> int:
    with conn() as c:
        return c.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]

if __name__ == '__main__':
    init()
    print(f'db ready at {DB_PATH}, jobs: {count_all()}')