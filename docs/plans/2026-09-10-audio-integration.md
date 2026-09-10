# Audio-генерация в gpux — дизайн интеграции (ACE-Step / Stable Audio / TTS)

> Новая модальность **audio** рядом с **image**. Механика свапа моделей —
> **та же**, что в [`2026-09-10-flux-dev-integration.md`](2026-09-10-flux-dev-integration.md):
> одна WanGP-сессия, `model_type` в задаче, `release_model()` + `reload_needed`,
> сериализация одним воркером, VRAM-гейт перед загрузкой, rollback при OOM,
> cooldown (reorder-window / min-residency / idle-unload). Здесь описаны только
> аудио-специфичные отличия. **Modality swap = model swap**, отдельного механизма нет.

---

## 0. Что реально есть в WanGP (сверено по исходникам)

`defaults/` содержит:

| Модель gpux (алиас) | WanGP `model_type` | architecture / handler | Тип | Выход |
|---|---|---|---|---|
| `ace_step` | `ace_step_v1_5_turbo_lm_4b` | `ace_step_v1_5` / `models/TTS/ace_step_handler.py` | музыка+вокал (Turbo, 8 шагов, LM 4B «Strong Think») | `.wav` |
| `ace_step_lite` | `ace_step_v1_5_turbo_lm_1_7b` | — | музыка, легче/быстрее | `.wav` |
| `stable_audio_sfx` | `stable_audio3_small_sfx` | `stable_audio3_small` / `models/TTS/stable_audio3_handler.py` | SFX/foley до 120 с | `.wav` |
| `stable_audio` | `stable_audio3_small` | — | музыка/атмосферы, лёгкая | `.wav` |
| `chatterbox` | `chatterbox` | `chatterbox` / `models/TTS/chatterbox_handler.py` | мультиязычный TTS + voice clone | `.wav` |
| `dramabox` | `dramabox_audio` | `dramabox_audio` / `models/ltx2/ltx_audio_tts_handler.py` | экспрессивный multi-speaker TTS | `.wav` |
| `qwen3_tts` | `qwen3_tts_base` | `qwen3_tts_base` / `models/TTS/qwen3_handler.py` | voice-clone TTS 1.7B | `.wav` |

Важное:
- Все аудио-архитектуры: `query_model_def` → **`image_outputs: False`**. WanGP пишет в `audio_save_path` = `outputs/` (тот же каталог, что картинки), формат **`.wav`** (`write_wav_file` / `save_audio_file`; muxed-кодек в конфиге `audio_output_codec`, для голого аудио — wav).
- **`stable_audio3_medium` требует Flash Attention 2**, а нода запущена с `--attention sdpa` → medium **не грузится**. Берём `small` / `small_sfx`.
- Аудио-задачи дают `progress`-события (фазы/шаги), **`preview` НЕ шлют** (нет картинки) — ветка `ev.kind == 'preview'` в воркере просто не срабатывает.

### Параметры по архитектурам (из defaults JSON)

| ключ WanGP | ACE-Step v1.5 | Stable Audio 3 | Chatterbox | DramaBox |
|---|---|---|---|---|
| `prompt` | текст песни (`[Verse]`/`[Chorus]`) | описание звука | текст для озвучки | `Speaker 1:` / `Speaker 2:` + ремарки |
| `alt_prompt` | стиль/жанр («Dreamy synth-pop…») | — | — | — |
| `negative_prompt` | — | `"poor quality, distorted, noisy"` | — | длинный дефолт |
| `duration_seconds` | 120 (макс ~240) | 8–60 (small до 120, sfx до 120) | 0 (авто) | 0 (авто, ×`duration_multiplier`) |
| `num_inference_steps` | 8 (turbo) | 8 | 0 | 30 |
| `audio_scale` | 0.5 | 0.9 | — | — |
| `guidance_scale` | 1.0 | 1.0 | — | 2.5 (+`audio_guidance_scale` 1.5) |
| `shift` / solver | `shift`1.0 `scheduler_type`"euler" | `sample_solver`"pingpong" | — | — |
| прочее | — | — | `model_mode`(язык), `temperature`0.8, `custom_settings.{exaggeration,pace}`, `audio_prompt_type`"A" (voice ref) | `multi_prompts_gen_type`"FG", `custom_settings.duration_multiplier` |

Voice-ref (клонирование голоса): `chatterbox`/`qwen3_tts` — референс-wav идёт как `audio_guide`, `apply_media_flag_defaults` сам проставит `audio_prompt_type` (флаг «A»).

---

## 1. Audio-API Spec

### 1.1. `POST /api/generate` — единый эндпоинт, поле `modality`

```jsonc
{
  "modality": "image | audio",              // default "image"
  "model": "ace_step | stable_audio_sfx | chatterbox | dramabox | qwen3_tts | flux2_klein_4b | flux_dev",

  // --- общее ---
  "prompt": "string",   // lyrics | описание | текст озвучки | Speaker-скрипт
  "seed": "int | null",

  // --- audio ---
  "style": "string",              // -> alt_prompt (жанр/стиль; ACE-Step, voice-design TTS)
  "negative_prompt": "string",    // Stable Audio / DramaBox / TTS
  "duration_s": "int",            // -> duration_seconds, clamp <= model_cost.max_duration_s
  "steps": "int",                 // -> num_inference_steps, clamp к steps_range
  "audio_scale": "float 0..1",    // per-model default
  "guidance_scale": "float",      // per-model default

  // --- TTS ---
  "language": "en|es|fr|de|ru|…",   // -> model_mode (chatterbox)
  "voice_ref": "data URL / base64 wav",  // -> INPUTS/<jid>.wav -> settings.audio_guide
  "tts": { "exaggeration": 0.5, "pace": 0.5, "temperature": 0.8 }  // -> custom_settings / temperature

  // --- image-поля (mode/init_image/resolution/denoise) — как сейчас, при modality:image ---
}
```

Сервер: `model` → `MODELS[m]`; проверка `MODELS[m].modality == req.modality` (иначе 400);
заполнение per-model дефолтов; clamp `duration_s`/`steps` → `warnings:[]` (никогда не 4xx за клампабельное).

Ответ на submit:
```jsonc
{ "id":"…", "node":"ms-7c75", "model":"ace_step", "modality":"audio",
  "queue":0, "busy":false, "swap":true, "eta_s":95, "warnings":["duration 300->240"] }
```

### 1.2. `GET /api/capabilities`

```jsonc
{
  "hostname": "ms-7c75",
  "modalities": ["image", "audio"],
  "models": ["flux2_klein_4b", "ace_step", "stable_audio_sfx", "chatterbox"],
  "loaded_model": "flux2_klein_4b",
  "loaded_modality": "image",
  "model_cost": {
    "ace_step": {
      "modality":"audio", "kind":"music", "wangp_model_type":"ace_step_v1_5_turbo_lm_4b",
      "output":"wav", "max_duration_s":240,
      "steps_range":[6,16], "steps_default":8,
      "audio_scale_default":0.5, "guidance_default":1.0,
      "load_time_s":50, "min_free_vram_gb":9, "typical_s_per_step":6,
      "params":["prompt(lyrics)","style","duration_s","audio_scale"]
    },
    "stable_audio_sfx": {
      "modality":"audio", "kind":"sfx", "wangp_model_type":"stable_audio3_small_sfx",
      "output":"wav", "max_duration_s":120,
      "steps_range":[6,16], "steps_default":8,
      "audio_scale_default":0.9, "guidance_default":1.0, "negative_prompt":true,
      "load_time_s":25, "min_free_vram_gb":4, "typical_s_per_step":2
    },
    "chatterbox": {
      "modality":"audio", "kind":"tts", "wangp_model_type":"chatterbox",
      "output":"wav", "languages":["en","es","fr","de","it","pt","pl","ru","zh","ja","…"],
      "voice_ref":true,
      "load_time_s":20, "min_free_vram_gb":3, "typical_job_s":8,
      "params":["prompt(text)","language","voice_ref","tts.{exaggeration,pace,temperature}"]
    }
  },
  "free_vram_gb": 14.6
}
```
`load_time_s / typical_s_per_step / swap_from` — самокалибруются из rolling-метрик (как в flux-dev §4).

### 1.3. `GET /api/status` — добавить (совместно с flux-dev)

`"loaded_model", "loaded_modality", "swapping", "swap_from", "swap_to", "swap_eta_s"`. `ready=false` пока `swapping`.

### 1.4. Отдача аудио

- Роутер `/api/image/{name}` (хардкод `image/jpeg`) → обобщить: **content-type по расширению** — `.wav`→`audio/wav`, `.mp3`→`audio/mpeg`, `.flac`→`audio/flac`, `.ogg`→`audio/ogg` + существующие image-типы. Оставить `/api/image` как алиас, канон — `/api/media/{name}`.
- **Range/seek**: `<audio>` в браузере скраббит через `Range`. wav на 4 мин ≈ 40 МБ. Сейчас `gpux-gallery` отдаёт `Response(content=bytes)` без Range → перевести `/api/image` gallery-сервиса на **`FileResponse`** (Starlette поддерживает Range), а роутер-прокси — **пробрасывать `Range` и стримить** `httpx.stream()` с `Content-Range`/`Accept-Ranges`.
- Скачивание: `<a download>` уже есть; корректный content-type → браузер сохранит `.wav`.

### 1.5. Sidecar / gallery-запись

Нода уже пишет `<file>.json`. Добавить в мету: `modality`, `kind` (`music|sfx|tts`), `duration_s`, `model`.
`gpux-gallery` и роутер отдают эти поля в `/api/gallery` как есть (пасс-тру).

---

## 2. Routing Logic

`pick_node(modality, model, feature)` уже есть — расширить:

1. `modality == "audio"` → требовать `"audio" in caps.modalities` **и** `model in caps.models`.
2. Per-model VRAM: `caps.model_cost[model].min_free_vram_gb <= h.free_vram_gb` (игра ест VRAM → ACE-Step 4B не влезет, хотя chatterbox бы влез).
3. Приоритет в ключе сортировки:
   `(loaded_model != model,           # 0 если свапа нет
     loaded_modality != req_modality, # 1 если меняется модальность (тяжелее: другой VAE/энкодер)
     busy, queue_len)`
   → выбираем ноду, где нужная модель уже резидентна; затем — где хотя бы та же модальность; затем любую подходящую.
4. `draft` (Horde overflow) + `modality:audio` → **400** (в Horde аудио не льём).
5. Единственная нода занята image-задачей → audio-запрос встаёт в очередь и свапнется после (группировка как в flux-dev §1.4).

`nodes.yaml` без изменений. `/api/status`/`/api/nodes` отдают `modalities` + `loaded_model` → фронт знает, что доступно прямо сейчас.

---

## 3. Implementation Plan — `node/serve.py`

Поверх мультимодельного дизайна flux-dev (`MODELS{}`, свап-логика). Аудио-дельта:

### 3.1. `node/config.py`

```python
MODELS = {
  "flux2_klein_4b": { "modality": "image", ... },      # как в flux-dev doc
  "flux_dev":       { "modality": "image", ... },
  "ace_step": {
    "modality": "audio", "kind": "music",
    "wangp_model_type": "ace_step_v1_5_turbo_lm_4b",
    "output_ext": "wav",
    "steps_range": (6, 16), "steps_default": 8,
    "max_duration_s": 240,
    "defaults": {"audio_scale": 0.5, "guidance_scale": 1.0, "shift": 1.0,
                 "scheduler_type": "euler"},
    "min_free_vram_gb": 9.0, "load_time_s": 50,
  },
  "stable_audio_sfx": {
    "modality": "audio", "kind": "sfx",
    "wangp_model_type": "stable_audio3_small_sfx",
    "output_ext": "wav",
    "steps_range": (6, 16), "steps_default": 8, "max_duration_s": 120,
    "defaults": {"audio_scale": 0.9, "guidance_scale": 1.0,
                 "negative_prompt": "poor quality, distorted, noisy",
                 "sample_solver": "pingpong"},
    "min_free_vram_gb": 4.0, "load_time_s": 25,
  },
  "chatterbox": {
    "modality": "audio", "kind": "tts",
    "wangp_model_type": "chatterbox", "output_ext": "wav",
    "languages": ["en","es","fr","de","it","pt","pl","ru","zh","ja"],
    "defaults": {"audio_prompt_type": "A", "model_mode": "en", "temperature": 0.8,
                 "num_inference_steps": 0, "video_length": 0,
                 "custom_settings": {"exaggeration": 0.5, "pace": 0.5}},
    "min_free_vram_gb": 3.0, "load_time_s": 20,
  },
}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg"}
# CAPABILITIES["modalities"] = sorted({m["modality"] for m in MODELS.values()})
```
`WAN2GP_ARGS` — оставить `--attention sdpa` (⇒ не включаем `stable_audio3_medium`).

### 3.2. `serve.py` — точечно

- **`api_generate`**: `req.modality ∈ {image,audio}`; `m = MODELS[req.model]`; если `m["modality"] != req.modality` → 400.
  - image-ветка — без изменений.
  - **audio-ветка**:
    ```python
    settings = {"model_type": m["wangp_model_type"], "prompt": req.prompt.strip(),
                **m["defaults"]}
    if req.style:            settings["alt_prompt"] = req.style
    if req.negative_prompt:  settings["negative_prompt"] = req.negative_prompt
    if req.duration_s:       settings["duration_seconds"] = min(req.duration_s, m["max_duration_s"])
    if req.steps:            settings["num_inference_steps"] = clamp(req.steps, *m["steps_range"])
    if req.audio_scale is not None: settings["audio_scale"] = req.audio_scale
    if req.guidance_scale is not None: settings["guidance_scale"] = req.guidance_scale
    if req.language:         settings["model_mode"] = req.language
    if req.tts:  settings.setdefault("custom_settings", {}).update(
                    {k: v for k, v in req.tts.items() if k in ("exaggeration","pace")})
                 if "temperature" in req.tts: settings["temperature"] = req.tts["temperature"]
    if req.voice_ref:
        p = _decode_audio_ref(req.voice_ref, jid)        # base64 wav -> INPUTS/<jid>.wav
        settings["audio_guide"] = str(p); input_file = p
    ```
  - `job["meta"] += {modality, kind: m["kind"], model: req.model, duration_s}`.
  - swap: `if req.model != state['loaded_model']: job["needs_swap"]=True` → ответ `swap:true, eta_s` (из `model_cost`/метрик).

- **`wan2gp_job_worker`**: свап-блок из flux-dev **без изменений** (release → empty_cache → 2s → VRAM-гейт vs `MODELS[want]["min_free_vram_gb"]` → load → rollback при OOM → нода жива на прежней модели).
  - event-loop: `ev.kind == 'preview'` для аудио не приходит — ок. Опц.: в начало аудио-джоба `broadcast({'type':'progress','id':jid,'phase':'audio','step':'0/'+steps,'pct':0})`.
  - результат: `files = result.generated_files` (`.wav`); sidecar-запись — как есть (мета уже с `modality/kind/duration_s`).
  - `push_to_gallery` / `_push_one`: MIME-мапа `+ wav/mp3/flac/ogg`.
  - `finally`: `input_file` (voice-ref wav) уже чистится тем же кодом, что и img2img `init_image`.

- **`api_status`**: `+loaded_model, +loaded_modality, +swapping, +swap_eta_s`.
- **`api_capabilities`**: `modalities`, `models`, `loaded_model`, `loaded_modality`, `model_cost` (§1.2).
- **`api_gallery`**: `ALLOWED_EXT |= AUDIO_EXT`; per-file `modality/kind/duration_s/model` из sidecar.
- **`api_image` → `api_media`** (алиас `/api/image`): content-type по расширению; `FileResponse` (Range для `<audio>`).

### 3.3. `gallery/serve.py` (niceguy)

- `IMG_EXT` → `MEDIA_EXT` (+ `wav/mp3/flac/ogg`).
- `/api/image/<name>`: отдавать через `FileResponse` (Range) + MIME по расширению.
- `/api/gallery`: пробросить `modality/kind/duration_s/model` из sidecar.
- `/ingest`: принимать аудио-расширения (проверка `Path(name).suffix in MEDIA_EXT`).

### 3.4. `router/serve.py`

- `GenRequest` `+= modality, style, duration_s, audio_scale, negative_prompt, language, voice_ref, tts:dict` (пролетят через `model_dump()`).
- `pick_node` — аудио-гейтинг (§2).
- `/api/image`-прокси → **проброс `Range`** + стрим `httpx.stream()` с `Content-Range`/`Accept-Ranges` (нужно для скраббинга аудио). В Plan-B режиме прокси идёт на `gpux-gallery`, там `FileResponse` уже даёт Range — роутеру достаточно форвардить заголовок и не буферизовать.
- `draft + modality:audio` → 400.

---

## 4. Frontend Integration (`web/index.html`)

### 4.1. Различение image / audio

- Верхний сегмент-контрол **Изображение / Аудио**. Переключение меняет панель параметров.
- Аудио → подвыбор: **Музыка (ACE-Step)** · **Звук/SFX (Stable Audio)** · **Речь (Chatterbox)**.
  - **Музыка**: textarea «Текст песни (`[Verse]`/`[Chorus]`)» + input «Стиль/жанр» + слайдер «Длительность, с» (10–240) + steps + `audio_scale`.
  - **SFX**: «Описание звука» + «Длительность, с» (1–30) + negative.
  - **Речь**: «Текст» + select языка + file «Голос-референс (wav)» + слайдеры exaggeration/pace.
- Body: `{ modality:"audio", model:"ace_step", prompt, style, duration_s, steps, audio_scale, ... }`.
- Доступность: из `/api/capabilities.modalities`; нет ноды с `audio` → вкладка Аудио disabled с тултипом.
- Свап-UX: как в flux-dev — `swap:true` → карточка «Загрузка аудио-модели… ~Ns» + отсчёт; SSE `model_swap`.
- Карточка джоба (аудио): прогресс-бар + шаг, **без превью-картинки**; по `job_done` — инлайн `<audio controls src=…>`.

### 4.2. Аудио в галерее

- `loadGallery()`: `if (f.modality === 'audio')` → плитка вместо `<img>`:
  ```html
  <div class="tile audio">
    <div class="wave">🎵 <span class="dur">2:00</span></div>
    <audio controls preload="none" src="${API}${f.url}"></audio>
  </div>
  ```
  бейдж `(i)` с промптом — уже есть, работает и тут (текст песни / описание).
- Лайтбокс для аудио: `<audio controls autoplay>` + промпт + кнопка «Скачать».
- Смешанная сетка (картинки + аудио-плитки) — норм, `grid` уже `auto-fill`.
- `<audio preload="none">` — не тянуть 40 МБ, пока не нажали play.

---

## 5. Deployment / rollout

1. **Набор моделей для ms-7c75**: `ace_step_v1_5_turbo_lm_4b` (музыка) + `stable_audio3_small_sfx` (SFX) + `chatterbox` (TTS). **Не** `stable_audio3_medium` (нужен FA2). ACE-Step 4B LM — самая тяжёлая (`min_free_vram_gb` ~9); сверить с live `free_vram_gb` (idle ~14.5 → влезает).
2. Пред-стейдж весов (WanGP качает по URL из `defaults/*.json`; текст-энкодер ACE-Step `acestep-5Hz-lm-4B` ~8 ГБ bf16 / ~4.5 ГБ int8). Warmup-сниппет на каждую модель (proxy unset!):
   ```bash
   cd /mnt/hdd/data/Wan2GP && unset http_proxy https_proxy ALL_PROXY
   /mnt/hdd/data/wan2gp_env/bin/python -c "
   from shared.api import init; from pathlib import Path
   s = init(root=Path('.'), cli_args=['--attention','sdpa','--profile','4'])
   s.run_task({'model_type':'stable_audio3_small_sfx','prompt':'test whoosh',
               'duration_seconds':4,'num_inference_steps':8})
   "
   ```
3. Код: `config.py MODELS{}` + audio-ветка `api_generate` + `AUDIO_EXT` в фильтрах/MIME (нода + `gallery/serve.py` + роутер) + `_decode_audio_ref`. `py_compile` под `wan2gp_env`.
4. Деплой ноды → `/mnt/hdd/data/webui/`, рестарт (~60 с). `curl /api/capabilities` → `modalities:["image","audio"]`.
5. Смоук по возрастанию стоимости: **SFX 8 с** (лёгкая модель, свап с klein) → **TTS** (chatterbox, короткая фраза) → **музыка** (ACE-Step, 60–120 с). Наблюдать `model_swap` SSE, `swap_s`, `peak_vram`, `.wav` в галерее, воспроизведение через публичный URL со скраббингом.
6. Деплой `gpux-gallery` + роутер + фронт на niceguy. Вкладка Аудио end-to-end.
7. Профилинг: per-model `load_s`, `s_per_step`, `peak_vram_gb`, отношение размер-wav/длительность. Занести в `model_cost`.

---

## 6. Constraint — не перегружать GPU

- **Modality swap, никогда не co-resident.** Одна WanGP-сессия, `release_model()` перед загрузкой другого семейства. VRAM-гейт перед каждой загрузкой: если `free < MODELS[want].min_free_vram_gb` → **abort свапа**, остаёмся на текущей модели, джоб → `error "insufficient_vram"`, `serve.py` **не падает**.
- Аудио-модели легче flux-dev — пик VRAM выбранного набора заведомо < 16 ГБ.
- Тот же сериализованный воркер (одна очередь) → нет одновременного GPU-доступа.
- RAM: offload-store аудио-моделей меньше, чем у flux-dev, но ACE-Step 4B LM + audio-VAE замерить (`free -m` в профилинге). Пока резидентна аудио-модель — **глушить gpux-idle Horde-worker** (расширить `.farming`, как в flux-dev §1.6).
- Rollback при ошибке загрузки → возврат к последней рабочей модели, `job_error` клиенту, процесс жив.

---

## 7. Deliverables — чек-лист

- [x] **Updated Audio-API Spec** — §1 (единый `/api/generate` c `modality`, per-model поля lyrics/style/duration_s/audio_scale/voice_ref, `model_cost` в capabilities, wav + content-type + Range).
- [x] **Routing Logic** — §2 (гейт по `modalities`+`model`+`min_free_vram_gb`, ключ сортировки с приоритетом «модель уже резидентна» → «та же модальность», запрет draft+audio).
- [x] **Implementation Plan** — §3 (`MODELS{}` c аудио, audio-ветка `api_generate`, `AUDIO_EXT`/MIME/Range в ноде+gallery+router, свап-логика переиспользуется из flux-dev).
- Плюс: §4 фронт (Изображение/Аудио, `<audio>` в галерее и лайтбоксе), §5 rollout, §6 защита GPU.

**Зависимость:** §3 строится на мультимодельной инфраструктуре из
[`2026-09-10-flux-dev-integration.md`](2026-09-10-flux-dev-integration.md) (`MODELS{}`,
`loaded_model`, swap-воркер, VRAM-гейт, cooldown). Делать после/вместе с ней.
