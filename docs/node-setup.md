# Установка gpux-ноды на новую GPU-машину (Linux + NVIDIA)

Предполагается: Ubuntu 22.04/24.04, NVIDIA-драйвер стоит, машина видит роутер
по сети (tailscale или LAN).

## 1. WanGP + venv

```bash
# диск под модели: чем быстрее, тем лучше (NVMe предпочтительно)
git clone https://github.com/deepbeepmeep/Wan2GP.git /opt/Wan2GP   # или свой форк
cd /opt/Wan2GP
python3 -m venv /opt/wan2gp_env
source /opt/wan2gp_env/bin/activate

# torch под свою CUDA (пример для cu130):
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

ВАЖНО: если на машине настроен SOCKS/HTTP-прокси в переменных окружения —
pip и запуск ноды могут падать на httpx. node/run.sh уже чистит прокси-переменные.

## 2. gpux node

```bash
git clone https://github.com/tester-bcs/gpux.git /opt/gpux
# поправить пути в /opt/gpux/node/config.py:
#   WAN2GP_ROOT, VENV_ACTIVATE, LISTEN_PORT
cd /opt/gpux/node
bash run.sh    # тестовый запуск, должен поднять :8095
curl 127.0.0.1:8095/api/status   # {"ready": false ...} → через ~30-60с {"ready": true}
```

Первая генерация скачает модели с HuggingFace (десятки ГБ, качается один раз,
поддерживает докачку после прерываний). Разогнать можно заранее:

```bash
python3 test_wan2gp.py   # промпт с котом, 640x480 — качает и генерит
```

## 3. Горячие чекпоинты на быстрый диск (опционально)

После первой загрузки в `WAN2GP_ROOT/ckpts/` появятся модели. Чтобы init был
быстрым (~32s вместо ~47s), перенеси большие файлы на NVMe и верни симлинки:

```bash
HOT=/path/to/nvme/wan2gp_hot
mkdir -p $HOT
cp ckpts/flux-2-klein-4b_quanto_bf16_int8.safetensors $HOT/
cp ckpts/flux2_vae.safetensors $HOT/
cp -r ckpts/Qwen3 $HOT/
rm ckpts/flux-2-klein-4b_quanto_bf16_int8.safetensors ckpts/flux2_vae.safetensors
ln -s $HOT/flux-2-klein-4b_quanto_bf16_int8.safetensors ckpts/
ln -s $HOT/flux2_vae.safetensors ckpts/
ln -s $HOT/Qwen3/* ckpts/Qwen3/
```

## 4. systemd (автозапуск)

```ini
# /etc/systemd/system/gpux-node.service
[Unit]
Description=gpux node (WanGP generator)
After=network-online.target

[Service]
Type=simple
User=YOURUSER
WorkingDirectory=/opt/gpux/node
ExecStart=/opt/gpux/node/run.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now gpux-node
```

## 5. Регистрация на роутере

На машине роутера допиши в `nodes.yaml`:

```yaml
nodes:
  - name: <hostname>
    url: http://<tailnet-or-lan-ip>:8095
```

и перезапусти роутер. Нода появится в `/api/status` и начнёт получать работу.
