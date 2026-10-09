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

## Local annotation & reannotation

Build and correct the labeled dataset entirely on-host:

```bash
cd inference
python -m annotate.preannotate --source /path/to/frames --out annotations
python -m annotate.annotation_app            # http://127.0.0.1:8765
```

- `preannotate.py` runs the local YOLO model over an image directory (or video with `--every N`) and writes model pre-annotations to `annotations/predictions.json` plus initial YOLO labels in `annotations/labels/`.
- The annotator UI reviews and corrects boxes, class (10 road-anomaly classes), severity, confidence, depth/area estimates, and per-frame telemetry/environment metadata.
- `GET /api/export` (or the Export button) writes `annotations/export/` with YOLO `labels/`, per-frame C-DAC JSON payloads in `payloads/`, a combined `payloads.jsonl`, annotated frames in `annotated/`, and a COCO dataset in `coco/annotations.json` — ready for training and for client staging review.

Class registry: pothole, crack_transverse, crack_longitudinal, rutting, debris, waterlogging, missing_lane_marking, edge_break, manhole, road_barrier. Severity levels are suggested from estimated area (level_1 < 400 cm² < level_2 < 1500 cm² < level_3). False positives can be marked with an "Ignore" checkbox; they export to `labels_ignore/` and can be remapped to a background class for hard-negative mining (`--ignore-as-class`).

## Train & deploy loop

```bash
cd inference
python -m annotate.train_pipeline --annotations annotations --model best.pt --epochs 50
python -m annotate.train_pipeline --annotations annotations --model best.pt --epochs 50 --ignore-as-class --tensorrt
```

Reads `annotations/export/`, writes `data.yaml`, optionally merges ignore regions into class 10 (background), runs `yolo detect train`, exports ONNX, and with `--tensorrt` builds a TensorRT engine. Retrained weights land at `inference/best_retrained.pt`; update `MODEL_PATH` in `.env` and restart the systemd units to deploy.
