# Pothole Detection — Jetson Orin Nano

This repository is the Jetson-only runtime for direct road-camera pothole detection.

## What runs on the Orion/Orin Nano

- Direct CSI/MIPI camera capture through Jetson GStreamer.
- USB/V4L2 camera fallback.
- Local Ultralytics YOLO inference with JetPack-compatible PyTorch.
- Optional USB/UART GNSS receiver using NMEA GGA/RMC sentences.
- Annotated display or headless operation.
- Local SQLite event storage and saved images.
- Local FastAPI status/data service for LAN monitoring.

There is no phone app, laptop inference service, cloud inference, training pipeline, or x86 Docker runtime in this repository.

## Repository

```text
inference/
  best.pt                 trained pothole model
  camera.py               USB and CSI camera capture
  gnss.py                 GNSS/NMEA reader
  local_runner.py         camera -> inference -> storage loop
  server.py               local records/status/image API
  requirements-jetson.txt Jetson Python dependencies
  install_jetson.sh       Jetson setup script
  .env.example            runtime configuration template
  jetson/                 systemd service templates
benchmark_model.py        local-device benchmark
LICENSE
```

## Quick deployment

From the laptop, copy this complete folder to the Jetson, including `inference/best.pt`:

```bash
scp -r . JETSON_USER@JETSON_IP:/tmp/pothole-detection
```

On the Jetson:

```bash
cd /tmp/pothole-detection/inference
chmod +x install_jetson.sh
./install_jetson.sh
nano /opt/pothole-detection/inference/.env
sudo systemctl start pothole-runner@$USER pothole-api@$USER
```

Check services:

```bash
sudo systemctl status pothole-runner@$USER pothole-api@$USER
journalctl -u pothole-runner@$USER -f
curl http://localhost:8000/health
```

See `inference/README_ORION.md` for CSI/USB camera setup, GNSS configuration, dashboard, and troubleshooting.

## Runtime API

- `GET /health` — service and runner status.
- `GET /local-status` — current camera, FPS, latency, detection, and GNSS state.
- `GET /potholes` — saved events; requires `x-api-key`.
- `GET /uploads/<file>` — saved and latest annotated frames.

The runner performs all inference locally. The API does not accept remote image inference requests.
