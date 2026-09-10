# Flux.1 [dev] (quantized) в gpux — дизайн интеграции

> Тир **Quality**: тяжёлая модель для финальных рендеров, рядом с быстрым
> `flux2_klein_4b` (тир **Speed**). Одна нода, одна WanGP-сессия, горячий свап
> модели между тирами. Прозрачно для юзера: запрос «Quality» при загруженной
> «Speed» — система свапает модель и выполняет задачу, а не отдаёт 500.

---

## 0. Headline-блокер: RAM на ms-7c75

Замер на `ms-7c75` (2026-09-10):

```
RAM total 31G | used 16G | free 1G | buff/cache 18G | available 14G
Swap 8G / 8G  ← используется на 100%
VRAM 16G (usable ~15.5G)
```

Машина **уже под давлением по RAM** (Klein INT8 + WanGP + node + gpux-idle Horde-worker + десктоп), своп забит.

- **flux-dev INT8** (`--profile 4`, mmgp offload в pinned RAM) требует **~17–18 GB pinned RAM** под offload-store. На этой ноде — OOM/жёсткий своп-траш при загрузке.
- **flux-dev Q4 (GGUF)** требует **~11–12 GB pinned RAM**. Тоже на грани: `available` ~14G, но своп уже полный.

**Вывод:** в текущем железе тир Quality без изменений не поедет. Варианты по возрастанию стоимости:

| # | Вариант | Что даёт | Цена |
|---|---|---|---|
| A | **+RAM до 64 GB** на ms-7c75 | INT8 dev с `--profile 4` работает штатно | железо |
| B | **Q4 GGUF dev** + освободить RAM: пока dev резидентен — глушить gpux-idle Horde-worker; поднять swap до 32 GB (HDD) как страховку от OOM-kill на пике загрузки | работает, но своп klein↔dev идёт с диска (~2.5–3 мин) | своп-траш, медленный свап |
| C | **flux-dev на отдельной ноде** (mini-pc / будущая GPU-машина с ≥48 GB RAM) | ms-7c75 остаётся чистым Speed-нодом, роутер шлёт Quality на dev-ноду | нужна вторая нода |
| D | `--profile 3` (меньше offload, больше VRAM) | меньше RAM | dev-Q4 ~11 GB VRAM резидентно → OOM при >768² на 16 GB. Не рекомендуется |

**Рекомендация:** цель — **A** (или **C**). До этого — **B** в деградированном режиме: Q4, `max_resolution` для dev = 768×768, Horde-idle отключается на время резидентности dev.

Остальной дизайн ниже не зависит от выбора A/B/C — меняются только числа в `model_cost` и `--profile`.

---

## 1. Model Swap Strategy

### 1.1. Архитектура: одна сессия, свап `model_type`

Подтверждено по исходникам WanGP:

- `shared/api._RUNTIME` — процессный синглтон; вторую сессию/процесс поднимать нельзя (борьба за mmgp-буферы и CUDA-контекст, ×2 pinned RAM).
- `wgp.py:242 release_model()` + флаг `reload_needed` — WanGP **штатно переключает модель**: при `submit_task(settings)` с другим `model_type` предыдущая модель выгружается, новая грузится.
- GGUF поддержан: `shared.qtypes.gguf`, `quant_router.register_file_extension("gguf", ...)`.

Значит: нода держит **одну** WanGP-сессию, кладёт в задачу `model_type ∈ {flux2_klein_4b, flux}` (или кастомный GGUF-деф), WanGP свапает веса. Стоимость свапа = время выгрузки (~3–5 с) + загрузки новой модели (klein ~60 с; dev Q4 ~70–110 с; dev INT8 ~90–150 с) **+ смена текст-энкодера** (klein=Qwen3 INT8 ~1 GB → dev=T5-XXL INT8 ~4.7 GB).

### 1.2. Триггер: автоматический, по полю запроса

- Основной путь — **`quality: "speed" | "quality"`** в `POST /api/generate` (или явный `model`). **Никакого обязательного `/api/load_model`** — иначе клиент тащит на себе cooldown-логику, а требование «прозрачно для юзера» ломается.
- `POST /api/load_model` — **опциональный admin/warm-up** эндпоинт для пред-прогрева перед пачкой Quality-рендеров.

### 1.3. Состояние ноды

```python
state['loaded_model']        # какой model_type резидентен сейчас
state['swapping']            # bool — идёт свап
state['swap_to'] / ['swap_from'] / ['swap_eta_s']
state['last_used']           # {model_type: ts последнего джоба}
state['metrics']             # rolling: per-model load_s, s_per_step, peak_vram, swap_s
```

### 1.4. Поток в воркере (`wan2gp_job_worker`)

1. Взять джоб. `want = job['settings']['model_type']`.
2. **Группировка против трэша:** если `want != loaded_model`, но в очереди в пределах `SWAP_REORDER_WINDOW_S` (45 с) есть джоб под `loaded_model` — выполнить его вперёд (не бесконечно: окно ограничено по времени, чтобы Quality не голодал).
3. **Min-residency:** не свапать с модели младше `MODEL_MIN_RESIDENCY_S` (90 с), **если очередь это не форсит** (в очереди только другой тир).
4. Если свап нужен:
   - `broadcast({'type':'model_swap','id':jid,'from':loaded,'to':want,'eta_s':<из metrics>})`
   - `state.update(busy=True, swapping=True, swap_*=...)`, `ready→false`
   - `release_model()` → `torch.cuda.empty_cache()` → 2 с settle → **гейт по VRAM**: `free = mem_get_info()`; если `free < MODELS[want]['min_free_vram_gb']` → **abort свапа**, джоб → error `"insufficient_vram (game/render active?)"`, нода **остаётся на старой модели живой**.
   - загрузка `want` в `try/except`; при OOM/ошибке — **rollback**: перезагрузить `loaded_model`, джоб → error `"model_load_failed"`, нода жива.
   - при успехе: `loaded_model = want`, записать `swap_s` в metrics.
5. Выполнить джоб штатно. По завершении: `last_used[want] = now`.

### 1.5. Cooldown / anti-thrash — сводка

| Механизм | Значение по умолчанию | Смысл |
|---|---|---|
| `SWAP_REORDER_WINDOW_S` | 45 с | одиночный Quality среди Speed-потока не дёргает свап туда-обратно каждый джоб — консекутивные одномодельные джобы группируются |
| `MODEL_MIN_RESIDENCY_S` | 90 с | защита от пинг-понга при чередовании 1:1 |
| `MODEL_IDLE_UNLOAD_S` | 600 с | пустая очередь 10 мин + резидентна не-default модель → свап назад на `DEFAULT_MODEL` (освобождает RAM/VRAM под gpux-idle Horde) |
| холодный старт | `DEFAULT_MODEL = flux2_klein_4b` прогревается в `wan2gp_init_worker` | Speed-запросы мгновенные с первой секунды |

Метрика трэша — **свапов на 100 джобов** (`/api/metrics`). Если высокая при смешанной нагрузке — эскалация: выделенная dev-нода (вариант C) или окно для Quality-батчей.

### 1.6. gpux-idle взаимодействие

Расширить логику `.farming`: **не фармить Horde, пока `loaded_model == flux` (dev)** — dev+T5 съедают RAM/VRAM, которые Horde-worker'у не отдать. При свапе назад на klein/idle — фарм возобновляется.

---

## 2. VRAM / RAM Budgeting

### 2.1. Оценки (RTX 5060 Ti 16 GB, `--profile 4` = mmgp offload в pinned RAM)

| Компонент | `flux2_klein_4b` (сейчас) | `flux` dev INT8 (quanto) | `flux` dev Q4 (GGUF Q4_K_M) |
|---|---|---|---|
| Трансформер (диск) | ~4.0 GB | ~12 GB | ~6.5 GB |
| Текст-энкодер | Qwen3 INT8 ~1.0 GB | T5-XXL INT8 ~4.7 GB | T5-XXL INT8 ~4.7 GB |
| CLIP-L | — | ~0.25 GB | ~0.25 GB |
| VAE | ~0.3 GB | ~0.16 GB | ~0.16 GB |
| **Pinned RAM (offload-store)** | ~5–6 GB | **~17–18 GB** | **~11–12 GB** |
| **Peak VRAM @1024², profile 4** | ~3–4 GB (замер: idle 14.5 free → busy ~12.4 free) | **~9–12 GB** (оценка) | **~6–8 GB** (оценка) |
| Load cold | ~60 с (замер) | ~90–150 с | ~70–110 с |
| Swap klein→ (release+load+T5) | — | ~95–160 с | ~75–115 с |

> Числа dev — **оценки**. Точные — только замером на ноде (см. 2.3). Разброс дают: квант, `--profile`, разрешение, batch.

### 2.2. Жёсткие требования

- **RAM:** INT8 dev → нода ≥ **48 GB** (лучше 64). Q4 dev → ≥ **32 GB и своп ≥ 16 GB**, плюс отключение gpux-idle на время резидентности. ms-7c75 сейчас 31 GB / своп полон → см. §0.
- **VRAM:** INT8 dev @1024² на 16 GB — на грани; **@1536² → OOM**. Поэтому `MODELS['flux_dev']['max_resolution']` **ниже**, чем у klein: `1024x1024` (INT8) / `768x768` (Q4 в деградированном режиме §0-B).
- **Диск:** `/mnt/hdd/data` — 800 GB свободно, не проблема. Веса dev класть рядом с klein: klein лежит в `/home/avk/wan2gp_hot/` (symlink из `ckpts/`) — если это SSD, dev тоже туда (ускоряет свап); иначе `ckpts/` на HDD (+ к времени свапа).

### 2.3. Процедура замера (реальный deliverable вместо фейк-точных чисел)

```bash
# нода, dev-модель загружена, джоб не идёт:
python -c "import torch;f,t=torch.cuda.mem_get_info(0);print('free',f/2**30,'total',t/2**30)"
# затем джоб на целевом res/steps; каждые 2 с сэмплить free VRAM в фазах:
#   text_encode → inference → vae_decode
# записать: baseline_free, min_free_infer, min_free_vae
# VRAM_footprint = total - min_free_vae + 1 GB (safety)
# параллельно: `free -m` каждые 5 с → пик used RAM, рост swap
```
Занести замеры в `model_cost` (см. §3.2) и, если `peak_vram > ~13 GB` — понизить `max_resolution` / уйти на Q4.

### 2.4. Safe Zone — когда роутер считает ноду busy

- Нода: `ready = model_ready AND NOT swapping AND vram_ok AND NOT farming AND NOT offline`.
  → **во время свапа `ready=false, busy=true`**, роутер (`pick_node`) не шлёт новые джобы; уже стоящие в очереди ноды — остаются.
- `/api/status` отдаёт `swapping, swap_from, swap_to, swap_eta_s, loaded_model`.
- Роутер `pick_node(model=...)`:
  1. `model in caps.models` (как сейчас);
  2. **`caps.model_cost[model].min_free_vram_gb <= h.free_vram_gb`** (если игра ест VRAM — dev не влезет, хотя klein бы влез; klein-запрос при этом всё ещё маршрутизируется);
  3. **prefer** ноду с `caps.loaded_model == model` (в ключ сортировки — избегаем свапа).
- При диспатче джоба, которому нужен свап (`req.model != caps.loaded_model`): роутер **локально** метит ноду busy в `_health` на `swap_eta_s`, чтобы не задиспатчить второй джоб в окне 5-секундного health-TTL.

---

## 3. API Contract

### 3.1. `POST /api/generate` — расширение

```jsonc
{
  "prompt": "string (required)",
  "negative_prompt": "string (default '')",      // dev honors; klein/distilled ignore
  "mode": "txt2img | img2img (default txt2img)",
  "init_image": "data URL / base64 (img2img)",

  "quality": "speed | quality (default speed)",   // ← основной юзер-facing knob
  "model": "flux2_klein_4b | flux_dev",           // опц., advanced; перекрывает quality

  "resolution": "WxH (default 512x512; clamp к max_resolution модели)",
  "steps": "int (default per-model: klein 6 / dev 28)",
  "guidance_scale": "float (dev only; default 3.5; для klein принудительно 1)",
  "seed": "int | null"
}
```

Поведение сервера:

- `quality:"quality"` → `model=flux_dev`, `steps` default 28, `guidance_scale` default 3.5, `resolution` clamp к dev-max.
- `quality:"speed"` (или отсутствует) → `model=flux2_klein_4b`, `steps` default 6, `guidance_scale := 1` (distilled), klein-max.
- Явный `model` перекрывает `quality`.
- Все clamp'ы — на сервере; выход за диапазон → **clamp + `warnings:[]`** в ответе, **никогда не 4xx** за клампабельное значение.
- Модель не загружена → **всё равно enqueue**; нода свапает; ответ содержит `swap:true, eta_s`.

Ответ на submit:
```jsonc
{ "id":"…", "node":"ms-7c75", "model":"flux_dev", "queue":1, "busy":false,
  "swap": true, "eta_s": 140, "warnings": ["steps clamped 60->50"] }
```

### 3.2. `GET /api/capabilities` — расширение

```jsonc
{
  "backends": ["wangp"], "hostname": "ms-7c75",
  "modalities": ["image"], "features": ["img2img"],
  "vram_gb": 16, "ram_gb": 31,
  "loaded_model": "flux2_klein_4b",              // ← что резидентно сейчас
  "models": ["flux2_klein_4b", "flux_dev"],
  "model_cost": {
    "flux2_klein_4b": {
      "tier": "speed", "quant": "int8",
      "max_resolution": "1536x1152",
      "steps_range": [4, 12], "steps_default": 6,
      "guidance": { "fixed": 1 },
      "load_time_s": 60, "min_free_vram_gb": 5,
      "typical_s_per_step": 1.0
    },
    "flux_dev": {
      "tier": "quality", "quant": "gguf-q4_k_m",
      "max_resolution": "1024x1024",
      "steps_range": [20, 50], "steps_default": 28,
      "guidance": { "min": 1.5, "max": 6.0, "default": 3.5 },
      "load_time_s": 110, "min_free_vram_gb": 8,
      "typical_s_per_step": 2.4,
      "swap_from": { "flux2_klein_4b": 105 }      // замеренная стоимость свапа
    }
  },
  "free_vram_gb": 14.6
}
```

`load_time_s / typical_s_per_step / swap_from` — **самокалибрующиеся** из rolling-метрик (§4), не хардкод.

### 3.3. `GET /api/status` — добавить

`"loaded_model", "swapping", "swap_from", "swap_to", "swap_eta_s"`. `ready=false` пока `swapping`.

### 3.4. `POST /api/load_model {"model":"flux_dev"}` — новый, опциональный

Admin/warm-up. Если нода idle — синхронный свап; иначе ставит «load-джоб» в очередь. Ответ `{"ok":true,"eta_s":110}`. Не нужен для прозрачного пути — нужен для пред-прогрева перед батчем.

### 3.5. SSE — новое событие

```
data: {"type":"model_swap","id":"<jid>","from":"flux2_klein_4b","to":"flux_dev","eta_s":105}
```
Фронт показывает «Загрузка модели качества… ~2 мин» с обратным отсчётом вместо застывшего бара, затем обычные `progress`.

---

## 4. Performance Profiling

Писать в sidecar-мету джоба (`<file>.json`, уже есть) + rolling-агрегаты на `/api/metrics`:

| Метрика | Как снять |
|---|---|
| `model`, `quant` | из настроек джоба |
| `swap_occurred` (bool), `swap_s` | release+load время в воркере |
| `text_encode_s` | длительность фазы `encoding_text` (WanGP progress phase) |
| `steps`, `inference_s`, **`s_per_step`** = inference_s/steps | фаза `inference` |
| `vae_decode_s` | фаза `decoding` |
| `peak_vram_gb` | сэмпл `mem_get_info` во время джоба → `total - min_free` |
| `total_s`, `queue_wait_s` | таймстемпы enqueue/start/done |
| `init_time_s` | один раз на процесс, на первую загрузку каждой модели |

Агрегаты на `/api/metrics`: p50/p90 `s_per_step` по модели, **свапов на 100 джобов** (индикатор трэша), средний `queue_wait_s` отдельно для Quality и Speed.

---

## 5. Deployment Plan

### 5.1. Веса

WanGP качает по URL из `defaults/<model>.json`. `flux.1-dev` уже есть штатно:

`/mnt/hdd/data/Wan2GP/defaults/flux.json` → `"name":"Flux 1 Dev 12B"`, `"architecture":"flux"`, URLs bf16 + `quanto_bf16_int8`. **`model_type = "flux"`.**

Альтернативы того же семейства (drop-in, `architecture:"flux"`): `flux_srpo` («realism 3×», хорош под «финальные рендеры»), `flux_krea` (эстетика/фото).

**Для Q4 GGUF** — кастомный деф `/mnt/hdd/data/Wan2GP/defaults/flux_dev_q4.json`:
```json
{
  "model": {
    "name": "Flux 1 Dev 12B (Q4 GGUF)",
    "architecture": "flux",
    "description": "FLUX.1 dev GGUF Q4_K_M — тир Quality для gpux",
    "URLs": ["https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_K_M.gguf"]
  },
  "prompt": "", "resolution": "1024x1024",
  "num_inference_steps": 28, "guidance_scale": 3.5, "batch_size": 1
}
```
> Сверить точные ключи схемы с `flux.json` / `flux_dev_kontext.json` и `models/flux/flux_handler.py::query_model_def`; путь GGUF в WanGP — через `shared.qtypes.gguf` (расширение `.gguf` уже зарегистрировано в `quant_router`).

Текст-энкодер T5-XXL (`T5_xxl_1.1_enc_quanto_bf16_int8.safetensors`, ~4.7 GB) и CLIP-L WanGP скачает один раз при первом использовании. Сейчас в `ckpts/` только `Qwen3` (klein) — T5 подтянется.

**Куда физически:** klein лежит в `/home/avk/wan2gp_hot/` (symlink из `ckpts/`). Если это SSD — dev-веса туда же (свап быстрее). Иначе — стандартный `ckpts/` на HDD.

**Пред-стейджинг** (чтобы первый Quality не ждал 5–10 мин на скачивании 6 GB):
```bash
cd /mnt/hdd/data/Wan2GP && unset http_proxy https_proxy ALL_PROXY
/mnt/hdd/data/wan2gp_env/bin/python -c "
from shared.api import init; from pathlib import Path
s = init(root=Path('.'), cli_args=['--attention','sdpa','--profile','4'])
s.run_task({'model_type':'flux','prompt':'warmup','resolution':'512x512','num_inference_steps':4})
"
```

### 5.2. `node/config.py` — мультимодельность

```python
DEFAULT_MODEL = "flux2_klein_4b"        # прогревается на старте (Speed)

MODELS = {
    "flux2_klein_4b": {
        "tier": "speed",
        "wangp_model_type": "flux2_klein_4b",
        "steps_range": (4, 12), "steps_default": 6,
        "guidance": {"fixed": 1},
        "max_resolution": "1536x1152",
        "min_free_vram_gb": 5.0,
        "embedded_guidance_scale": 1,
    },
    "flux_dev": {
        "tier": "quality",
        "wangp_model_type": "flux",          # или "flux_dev_q4" для GGUF
        "steps_range": (20, 50), "steps_default": 28,
        "guidance": {"min": 1.5, "max": 6.0, "default": 3.5},
        "max_resolution": "1024x1024",        # ← ниже klein; для Q4-деград-режима 768x768
        "min_free_vram_gb": 8.0,
        "embedded_guidance_scale": 3.5,
    },
}

# свап-поведение
MODEL_MIN_RESIDENCY_S = 90
MODEL_IDLE_UNLOAD_S   = 600
SWAP_REORDER_WINDOW_S = 45

WAN2GP_ARGS = ["--attention", "sdpa", "--profile", "4"]   # profile 3 только если A/C и мало RAM

CAPABILITIES = {
    "backends": ["wangp"], "modalities": ["image"], "features": ["img2img"],
    "vram_gb": 16, "ram_gb": 31, "hostname": "ms-7c75",
    "models": list(MODELS),
    # model_cost собирается в serve.py из MODELS + rolling-метрик
}
```

`MODEL_TYPE` (старое одиночное поле) — оставить алиасом `= DEFAULT_MODEL` для обратной совместимости, пока роутер/фронт не переедут на `models`.

### 5.3. `node/serve.py` — изменения (сводка, не полный код)

- state: `loaded_model, swapping, swap_*, last_used, metrics`.
- `wan2gp_init_worker`: после `init()` — тёплая загрузка `DEFAULT_MODEL` (крошечный throwaway-таск), `loaded_model = DEFAULT_MODEL`.
- `api_generate`: `quality`→`model`; валидация по `MODELS`; per-model клампы `steps`/`resolution`/`guidance` + `warnings`; в `job['settings']` — `model_type = MODELS[m]['wangp_model_type']`, `embedded_guidance_scale`, (dev) `guidance_scale`, `negative_prompt`; если `model != loaded_model` → `job['needs_swap']=True`, ответ `swap:true, eta_s`.
- `wan2gp_job_worker`: группировка (§1.4.2) → min-residency (§1.4.3) → свап с VRAM-гейтом и rollback (§1.4.4) → джоб → метрики по фазам.
- `api_status`: свап-поля; `ready` учитывает `not swapping`.
- `api_capabilities`: `loaded_model`, `model_cost` (config ⊕ измеренные rolling-средние).
- новый тред «idle-unload»: очередь пуста > `MODEL_IDLE_UNLOAD_S` и `loaded_model != DEFAULT_MODEL` → свап назад.
- `.farming`-логика: не фармить при `loaded_model == flux_dev`.

### 5.4. `router/serve.py` — изменения

- `GenRequest`: `+quality, +guidance_scale, +negative_prompt` (пролетят через `model_dump()`).
- `pick_node(model=...)`: гейт по `model_cost[model].min_free_vram_gb`; в ключ сортировки — бонус за `loaded_model == model`.
- при диспатче со свапом — локально пометить ноду busy на `swap_eta_s`.
- ETA (`eta_s`, `swap`) пробросить во фронт как есть.

### 5.5. Фронт (`web/index.html`)

- Сегмент-контрол **Speed / Quality** → `quality` в body. Подсказка у Quality: «≈2–3 мин, до 1024². Первый рендер грузит модель ~2 мин».
- Ответ `swap:true` → карточка джоба: «Загрузка модели качества…» + обратный отсчёт `eta_s` вместо 0%-бара; SSE `model_swap` — то же.
- Диапазоны слайдеров `steps` и новый слайдер `guidance` — из `/api/capabilities.model_cost[model]` (динамически по выбранному тиру).
- Бейдж модели на плитках галереи (уже есть) теперь осмысленно различает klein/dev.

### 5.6. Порядок раскатки

1. **`free -g` + `swapon` на ноде** → решить A/B/C и quant (INT8 vs Q4). Без этого дальше нет смысла (см. §0).
2. Положить деф (`flux.json` штатный или кастомный `flux_dev_q4.json`); пред-стейдж весов сниппетом (proxy unset!).
3. `config.py MODELS` + свап-логика в `serve.py` в репо; `py_compile` под `wan2gp_env`.
4. Деплой в `/mnt/hdd/data/webui/`, рестарт ноды (~60 с). Проверить `/api/capabilities` → оба модели, `loaded_model=flux2_klein_4b`.
5. Смоук: Speed-джоб (без свапа) → Quality-джоб (наблюдать `model_swap` SSE, `swap_s`, завершение, пуш в gallery с `model:flux_dev` в сайдкаре) → снова Speed (свап назад). Следить за трэшем и ростом свопа.
6. Деплой router + фронт на niceguy; Speed/Quality-тумблер end-to-end через `studio.51-89-98-172.sslip.io`.
7. Профилинг: 10 Quality-рендеров → записать `s_per_step`, `swap_s`, `peak_vram_gb`, пик RAM. Занести в `model_cost`. Если `peak_vram > ~13 GB` или RAM в свопе — понизить `max_resolution` / уйти на Q4 / `--profile`.

### 5.7. Риски

| Риск | Митигация |
|---|---|
| OOM на свапе (обе модели кратко резидентны) | `release_model()` → `empty_cache()` → 2 с → VRAM-гейт **перед** загрузкой; при нехватке — abort + остаться на старой |
| RAM-траш / OOM-kill на пике загрузки dev | Q4; своп 32 GB; глушить gpux-idle при dev; в идеале — +RAM (A) или отдельная нода (C) |
| Пинг-понг свапов при смешанной нагрузке | группировка + `MODEL_MIN_RESIDENCY_S`; при устойчивом трэше — выделенная dev-нода / окно для Quality-батчей |
| Латентность первого Quality | пред-стейдж весов; `/api/load_model` warm-up перед батчем |
| Свап T5↔Qwen3 незаметно раздувает время свапа | включено в замеряемый `swap_from` |
| `1536²` на dev → OOM | `MODELS['flux_dev']['max_resolution']` жёстко ниже klein; клампится на сервере с `warnings` |

---

## 6. Deliverables — чек-лист

- [x] **Model Swap Strategy** — §1 (одна сессия, авто-триггер по `quality`/`model`, VRAM-гейт + rollback, cooldown: reorder-window + min-residency + idle-unload).
- [x] **Updated API Spec** — §3 (`quality: speed|quality`, per-model клампы+`warnings`, `swap/eta_s` в ответе, `model_cost` в capabilities, `model_swap` SSE, опц. `/api/load_model`).
- [x] **Deployment Plan** — §5 (веса в `defaults/` + `ckpts/`/`wan2gp_hot`, пред-стейдж, `config.py MODELS`, правки node/router/front, порядок раскатки, риски).
- [x] **VRAM/RAM Budgeting** — §2 + §0 (таблица оценок, процедура замера, safe-zone для роутера).
- [x] **Performance Profiling** — §4 (per-phase метрики, `s_per_step`, свапов/100 джобов, самокалибровка `model_cost`).

**Блокер к старту:** §0 — на ms-7c75 (31 GB RAM, своп полон) тир Quality без +RAM / Q4-деград-режима / отдельной ноды не поедет. Решение по A/B/C — до кода.
