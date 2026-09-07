# Copilot Instructions — Jetson Orin Nano Pothole Detection

## Project identity

This repository is the **Jetson Orin Nano deployment** for direct road-camera pothole detection. It is not a laptop, phone, Expo, React, training, or cloud-inference project.

All runtime work must remain on the Jetson:

```text
CSI/USB camera → local capture → local YOLO inference → annotated frame
                                      ├→ SQLite + saved image events
                                      ├→ GNSS coordinates
                                      └→ FastAPI + embedded dashboard
```

The Jetson is both the camera host and the inference host. Do not reintroduce a phone upload workflow or a remote inference server.

## Repository layout

- `inference/best.pt` — trained pothole model; required for deployment.
- `inference/camera.py` — USB/V4L2 and CSI/GStreamer camera opening.
- `inference/gnss.py` — dependency-light NMEA GGA/RMC parser and background GNSS reader.
- `inference/local_runner.py` — primary camera-to-inference-to-storage process.
- `inference/server.py` — local FastAPI API and dashboard host; it does not accept remote image inference.
- `inference/dashboard/index.html` — embedded dashboard with Leaflet/OpenStreetMap GIS map.
- `inference/install_jetson.sh` — Jetson setup script.
- `inference/jetson/` — user-specific systemd service templates.
- `inference/requirements-jetson.txt` — non-PyTorch Jetson dependencies.
- `inference/.env.example` — deployment configuration template.
- `benchmark_model.py` — local-device benchmark with CUDA synchronization.
- `inference/tests/` — hardware-independent helper tests.

Do not add mobile clients, React/Node dashboards, training pipelines, x86 CUDA Dockerfiles, laptop-specific runtime code, or cloud inference unless the project owner explicitly changes the deployment target.

## Runtime rules

1. **PyTorch first.** Use JetPack-compatible NVIDIA PyTorch/torchvision already installed on the Orin Nano. Never install generic x86 or PyPI CUDA wheels over the JetPack build.
2. **TensorRT is optional.** Add TensorRT only after PyTorch works and real-camera accuracy is verified. TensorRT engines are hardware, JetPack, and TensorRT-version specific.
3. **CSI is preferred.** Use `nvarguscamerasrc` through GStreamer for CSI/MIPI cameras. USB cameras use V4L2/OpenCV.
4. **Keep latency bounded.** Camera pipelines must drop stale frames (`appsink drop=true max-buffers=1`) rather than building an unbounded queue.
5. **GNSS is optional at runtime.** A missing receiver or GPS fix must not stop detection. Save null coordinates when there is no valid fix.
6. **Avoid duplicate events.** Detection event writes use the configurable `SAVE_COOLDOWN`; do not save every frame by default.
7. **Headless support is required.** Systemd starts the runner with `--no-display`. Do not require an X display for inference or storage.
8. **Do not commit secrets or generated data.** Keep API keys in `.env`, never commit `.env`, databases, uploads, status JSON, logs, or generated engine files.
9. **Use absolute/configurable paths.** Runtime paths are configured through environment variables such as `MODEL_PATH`, `DB_PATH`, `UPLOADS_DIR`, and `STATUS_PATH`.
10. **Preserve the API/dashboard contract.** Keep `/health`, `/local-status`, `/potholes`, `/uploads/...`, and `/` dashboard behavior compatible unless deliberately versioning the API.

## Future TensorRT roadmap

TensorRT must be tried in a later optimization phase, not assumed to work before the PyTorch baseline is stable.

1. Capture representative road-camera frames and record the PyTorch baseline for accuracy, end-to-end latency, inference latency, FPS, memory, and temperature on the actual Orin Nano.
2. Export `inference/best.pt` to a TensorRT FP16 engine on the target Jetson or a demonstrably compatible JetPack environment.
3. Run the same validation frames through PyTorch and TensorRT. Check missed potholes, false positives, bounding boxes, confidence behavior, and class mapping before accepting the engine.
4. Compare sustained real-camera performance, not synthetic or theoretical projections. Include startup time, dropped frames, thermal behavior, and storage/API behavior.
5. Only after FP16 is accurate and stable, investigate INT8 with a representative calibration dataset. Document calibration details and repeat the accuracy comparison.
6. Keep `best.pt` as the fallback/debug model and make the TensorRT engine selectable through configuration. Never commit a generated engine unless its JetPack, Jetson model, TensorRT version, precision, and calibration provenance are documented.

Do not claim TensorRT speedups or accuracy until the tests have been run on the physical Orin Nano.

## Camera configuration

Important environment variables:

- `CAMERA_TYPE=csi` or `usb` — default deployment is CSI.
- `CSI_SENSOR_ID=0` — CSI sensor index.
- `CAMERA_DEVICE=0` — USB index or device path.
- `CAMERA_WIDTH`, `CAMERA_HEIGHT`, `CAMERA_FPS` — camera capture settings.
- `INFERENCE_WIDTH`, `INFERENCE_HEIGHT` — CSI output dimensions.
- `IMAGE_SIZE=640` — YOLO inference size.
- `CONFIDENCE=0.25` — detection threshold.
- `DISPLAY=false` — use headless mode for systemd.

CSI pipelines depend on the exact camera sensor and JetPack version. Do not assume every CSI camera supports every resolution/FPS combination.

## GNSS configuration

- `GNSS_PORT=/dev/ttyUSB0` or `/dev/ttyACM0`.
- `GNSS_BAUD=9600` unless the receiver specifies another rate.
- Supported sentences are standard NMEA GGA and RMC.
- The deployment user normally needs `dialout` permission.

## Dashboard and GIS

FastAPI serves the dashboard directly at:

```text
http://JETSON_IP:8000/
```

The dashboard displays runner state, FPS, latency, GNSS fix, latest annotated frame, saved events, and GPS markers. Leaflet renders the map and OpenStreetMap provides tiles. Inference and storage remain local; only map tiles require internet access. Without internet, coordinates and event records must still work.

The `/potholes` endpoint requires the `x-api-key` header. `/health` and `/local-status` are public for monitoring. The API key belongs in `.env` and must never be hardcoded into source or dashboard files.

## Deployment commands

On the Jetson, install from the copied repository:

```bash
cd inference
chmod +x install_jetson.sh
./install_jetson.sh
nano /opt/pothole-detection/inference/.env
sudo systemctl start pothole-runner@$USER pothole-api@$USER
```

Useful checks:

```bash
sudo systemctl status pothole-runner@$USER pothole-api@$USER
journalctl -u pothole-runner@$USER -f
curl http://localhost:8000/health
```

Foreground CSI test:

```bash
python3 local_runner.py --camera-type csi --display
```

Foreground USB test:

```bash
python3 local_runner.py --camera-type usb --camera-device 0 --display
```

## Development and verification

Before completing changes:

- Run Python compilation for changed modules.
- Run hardware-independent tests under `inference/tests/`.
- Check `get_errors`/editor diagnostics.
- Do not claim camera, GNSS, CUDA, FPS, thermal, or TensorRT results without running on the actual Orin Nano.
- Keep benchmark results measured on the target device; do not use theoretical Jetson projections.
- Validate both display and `--no-display` paths when changing the runner.

Hardware testing is required for final acceptance of camera pipelines, GNSS serial access, CUDA performance, and systemd startup.
