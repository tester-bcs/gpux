# gpux Integrations — Plan (capabilities → учёт → Horde overflow → idle-mode)

> **For Hermes:** использовать subagent-driven-development для исполнения таска за таском.

**Goal:** превратить gpux из «роутера по принципу первая-свободная» в маршрутизатор
с knowledge о нодах (capabilities), учётом потребления и эластичным overflow в
AI Horde.

**Architecture:** нода отдаёт `/api/capabilities` (движки, модели, VRAM); роутер
хранит снапшоты и маршрутизирует по типу задачи; в sqlite копится учёт юзер/нода/
шаги; Horde подключается как backend-тип `horde` (async submit/poll), а нода
получает опциональный idle-режим с horde-worker-reGen.

**Tech Stack:** FastAPI, httpx, sqlite3 (stdlib), pyyaml; WanGP API на нодах.

**Референс анализа:** `docs/integrations.md`

---

## Sprint 1: Node Protocol v1 + маршрутизация по capabilities (1 вечер)

### Task 1.1: `/api/capabilities` на ноде

**What:** нода отдаёт статическое описание себя.

**How:**
1. В `node/config.py` добавить поле:
```python
CAPABILITIES = {
    "backends": ["wangp"],
    "models": ["flux2_klein_4b"],
    "modalities": ["image"],          # позже: ["image", "video", "audio"]
    "vram_gb": 16,
    "max_resolution": "1536x1152",
}
```
2. В `node/serve.py` добавить эндпоинт (рядом с `/api/status`):
```python
@app.get("/api/capabilities")
async def api_capabilities():
    return config.CAPABILITIES
```

**Criteria of success:**
- [ ] `curl node:8095/api/capabilities` возвращает JSON с backends/models/modalities
- [ ] остальные эндпоинты не сломаны (`/api/status` отвечает)

### Task 1.2: Роутер кэширует capabilities

**What:** при старте и раз в HEALTH_TTL роутер тянет capabilities нод.

**How:**
1. В `router/serve.py` в `refresh_health()` добавить запрос:
```python
r2 = await client.get(n['url'] + '/api/capabilities')
caps = r2.json() if r2.status_code == 200 else {}
```
и положить в `_health[name]['caps']`.
2. В `/api/nodes` отдавать caps наружу.

**Criteria of success:**
- [ ] `curl router:8096/api/nodes` показывает capabilities ноды
- [ ] нода без capabilities (старая версия) не валит роутер (caps = {})

### Task 1.3: Маршрутизация по modality/model

**What:** `POST /api/generate` принимает опциональные `modality` и `model`;
роутер выбирает ноду, которая их умеет.

**How:**
1. `GenRequest`: добавить `modality: str = "image"`, `model: str | None = None`.
2. В `pick_node()` фильтр: `req_model in caps['models']` (если задан),
   `req_modality in caps['modalities']`; затем прежняя логика (free → busy).
3. Если ни одна нода не умеет — 503 с осмысленным detail.

**Criteria of success:**
- [ ] generate без полей ведёт себя как раньше (regression)
- [ ] `{"model": "несуществует"}` → 503 "no GPU nodes can serve model ..."
- [ ] `{"modality": "video"}` при отсутствии видео-нод → 503

### Task 1.4: Коммит + деплой

- [ ] `git commit` после каждого таска
- [ ] деплой обновлённых node/serve.py+config.py на ms-7c75, router/serve.py на vps-ru
- [ ] E2E через nginx: generate → done (как раньше), `api/nodes` показывает caps

**Sprint 1 estimate:** 2-3 часа

---

## Sprint 2: Учёт потребления в sqlite (1 вечер)

### Task 2.1: Схема и миграция

**How:**
1. `router/db.py`:
```python
import sqlite3, time
DB = HERE / 'usage.db'
def conn(): 
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c
def init():
    with conn() as c:
        c.executescript('''CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY, ts REAL, user TEXT DEFAULT 'anon',
  node TEXT, model TEXT, resolution TEXT, steps INT, seed INT,
  status TEXT, total_s REAL, kudos_est REAL);''')
```
2. Вызов `init()` при старте роутера.

### Task 2.2: Запись задач

**How:** в `/api/generate` (после успешного submit) — INSERT; в `/api/jobs/{id}`
при статусе done/error — UPDATE.

### Task 2.3: Эндпоинт статистики

- [ ] `GET /api/usage?limit=50` — последние задачи
- [ ] `GET /api/usage/summary` — по юзерам: задач, суммарные шаги, GPU-секунды

**Criteria of success:**
- [ ] после E2E-генерации в usage.db есть строка с node/model/steps/total_s
- [ ] `/api/usage/summary` считает по юзерам
- [ ] файл usage.db в .gitignore

**Sprint 2 estimate:** 2 часа

---

## Sprint 3: Horde-adapter (overflow для черновиков) (1-2 дня)

### Task 3.1: Конфиг Horde в роутере

**How:** `router/nodes.yaml` дополнить секцией:
```yaml
horde:
  enabled: false
  api_url: https://aihorde.net/api/v2
  api_key: "0000000000"     # аноним; позже — зарегистрированный ключ
  models: ["AlbedoBase XL (SDXL)"]
```
- [ ] config-класс читает секцию; disabled по умолчанию

### Task 3.2: Horde-клиент (async submit/poll)

**How:** `router/horde.py`:
- `submit(prompt, steps, ...)` → `POST /generate/async` → id
- `poll(id)` → `GET /generate/check/{id}` до done
- `result(id)` → `GET /generate/status/{id}` → base64 картинки
- сохранить артефакт в `outputs/` ноды-прокси? Нет: в `router/artifacts/`,
  отдавать через собственный `/api/image/...`
- все запросы с заголовком `apikey`; `Client-Agent: gpux:0.1:tester-bcs`

### Task 3.3: Маршрутизация overflow

**What:** если в `/api/generate` пришёл `draft: true` (или все ноды заняты
больше N секунд) и horde.enabled — задача уходит в Horde.

**Criteria of success:**
- [ ] draft-задача при выключенном horde → 503 "draft overflow disabled"
- [ ] draft-задача при включённом → задача доезжает, картинка в галерее
  (помечена node="aihorde")
- [ ] в usage.db видно node='aihorde'

### Task 3.4: Дисклеймер приватности

- [ ] в UI чекбокс «черновик (через публичную сеть, промпт уйдёт наружу)» —
  выключен по умолчанию
- [ ] в docs/integrations.md — ссылка на раздел Риски (уже есть)

**Sprint 3 estimate:** 6-10 часов

---

## Sprint 4: Idle-mode ноды (horde-worker-reGen) (1 день, опционально)

### Task 4.1: Скрипт-надзиратель

**How:** `node/idle_worker.sh` + systemd-таймер:
- idle-детект: `/api/status` ноды `!busy && queue==0` дольше IDLE_MIN (30m)
- поднять `horde-worker-reGen` (отдельный venv, SDXL-модель)
- при появлении задачи в очереди (poll /api/status) — SIGTERM воркеру,
  дождаться завершения VRAM, вернуться в тёплый режим
- kudos-аккаунт: зарегистрировать на aihorde.net, ключ в env

**Criteria of success:**
- [ ] простой >30m → воркер Horde активен (видно в их UI)
- [ ] юзерская задача → воркер остановлен <60s, генерация тёплая
- [ ] kudos копятся на аккаунте

### Task 4.2: Документация режима

- [ ] docs/node-setup.md — раздел «Idle-mode (фарм kudos)», риски приватности

**Sprint 4 estimate:** 4-6 часов

---

## Что NOT делать (anti-scope)

- [ ] НЕ поднимать собственный AI Horde сервер (заменит наш роутер)
- [ ] НЕ встраивать ComfyUI-Distributed (двойной мультиплексор)
- [ ] НЕ требовать одинаковые модели на нодах (анти-паттерн A1111-distributed)
- [ ] НЕ конвертировать kudos в токены prolog-bc (ToS Horde)
- [ ] НЕ слать промпты в Horde без явного draft-флага юзера
- [ ] НЕ трогатьprm/mtproto/niceguy-сервисы при деплоях

## Timing estimate

- Sprint 1: 2-3 часа
- Sprint 2: 2 часа
- Sprint 3: 6-10 часов
- Sprint 4: 4-6 часов (опционально)
- Итого ядро (S1-S3): ~1.5 рабочих дня

---

# Запрос на approve
- [ ] Approve — исполнить S1-S3
- [ ] Approve — исполнить S1-S4 (включая idle-фарм)
- [ ] Rework (что поменять)
