# gpux

Прототип мультиплексора GPU-генерации изображений: N машин с GPU (ноды) +
роутер, который раздаёт запросы пользователей свободной ноде.

**Статус:** рабочий прототип, 1 нода (ms-7c75, RTX 5060 Ti) + роутер (vps-ru).
Дальше — подключение новых GPU-машин и честный планировщик (см. Roadmap).

## Архитектура

```
пользователь
   │  https://45-138-157-250.sslip.io/studio/
   ▼
vps-ru: nginx (basic auth) ──► gpux router :8096        ← мультиплексор
                                    │  health-check нод, выбор свободной
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
              node ms-7c75    node <future>    node <future>
              (WanGP API,     (WanGP API,      (WanGP API,
               RTX 5060 Ti)    GPU ...)         GPU ...)
```

- **node/** — агент для GPU-машины: держит WanGP-сессию загруженной, очередь
  задач, SSE-прогресс с живыми превью, галерею. Один и тот же код на любой ноде.
- **router/** — мультиплексор: реестр нод в `nodes.yaml`, health-check
  (`/api/status` раз в 5с), выбор свободной ноды (ready && !busy && queue==0),
  sticky routing по job id, склейка галерей всех нод.
- **web/** — фронт (тёмный UI в стиле reve/nanobanana), общается только с
  роутером, нод не знает.
- **deploy/** — установка ноды, nginx-конфиг vps-ru.

## Текущий деплой

| Что | Где | Как |
|---|---|---|
| Нода-генератор | ms-7c75:8095 | `node/run.sh` (venv `/mnt/hdd/data/wan2gp_env`) |
| Роутер | vps-ru:8096 | systemd `gpux-router.service` |
| Фронт+auth | vps-ru nginx | `https://45-138-157-250.sslip.io/studio/` |

Развёрнутые копии живут в `/mnt/hdd/data/webui/` (нода) и
`/opt/gpux-router/` (vps-ru) — этот репозиторий источник истины.

## Запуск

### Нода (GPU-машина)
```bash
# 1. Установи WanGP (см. https://github.com/deepbeepmeep/Wan2GP) в venv
# 2. Поправь node/config.py под пути машины
# 3. Запусти:
bash node/run.sh            # слушает :8095
```

### Роутер (любая машина с python3.12, видит ноды по сети)
```bash
pip install fastapi uvicorn httpx pyyaml
cp router/nodes.yaml.dist router/nodes.yaml   # впиши свои ноды
python3 router/serve.py                       # слушает 127.0.0.1:8096
```

### Проверка ноды вручную
```bash
curl node:8095/api/status
curl node:8095/api/gallery | head
```

## API (роутер == нода, совместимы)

| Endpoint | Что |
|---|---|
| `GET /api/status` | ready/busy/queue (+ список нод на роутере) |
| `POST /api/generate` | `{prompt, resolution, steps, seed?}` → `{id, node}` |
| `GET /api/jobs/{id}` | статус задачи |
| `GET /api/events` | SSE: progress/preview/job_done/job_error |
| `GET /api/gallery` | последние генерации (на роутере — склейка всех нод) |
| `GET /api/image/{name}` | картинка из outputs |

## Добавить новую GPU-машину

1. Установи WanGP + venv (инструкция в docs/node-setup.md)
2. Залей `node/` на машину, поправь `node/config.py`
3. Запусти `bash node/run.sh` (или создай systemd-юнит)
4. На роутере добавь машину в `nodes.yaml`, перезапусти роутер
5. Нода появится в `/api/status` и начнёт принимать работу автоматически

## Roadmap: мультиплексор

Сейчас (прототип): запрос → первая свободная нода. Модель загружается на ноде
один раз и живёт в VRAM; занятая нода пропускается.

Дальше:
1. **Registration API** — ноды сами регистрируются на роутере (heartbeat),
   вместо ручного nodes.yaml
2. **Планировщик с весами** — VRAM/RAM/скорость ноды, приоритеты юзеров
3. **Очередь на роутере** — если все ноды заняты, очередь задач на роутере
   с позициями и отменой
4. **Мульти-модель** — роутер знает, какая нода какую модель держит
   (flux / wan video / tts), роутинг по типу задачи
5. **Аутентификация юзеров** — токены вместо basic auth, лимиты на юзера
6. **Учёт** — кто, когда, сколько шагов, на какой ноде (sqlite на роутере)
7. **Вытеснение** — если нода ушла в оффлайн посреди задачи, задача
   перезапускается на другой (seed фиксирован — результат воспроизводим)
