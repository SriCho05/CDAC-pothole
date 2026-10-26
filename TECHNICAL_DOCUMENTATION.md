# CDAC Pothole Detection — Technical Documentation & Setup Guide

> For the testing team. Covers what the project is, how to start the complete dashboard, and how to run the live inference pipeline.

---

## 1. What this project is

A pothole / road-anomaly detection system running **entirely on an NVIDIA Jetson Orin Nano**. A camera feed is processed on-device by a TensorRT INT8 engine, anomalies are boxed and labelled, and events (with snapshots) are stored in a local SQLite database. A single FastAPI app serves a browser dashboard with a live MJPEG feed, a GIS map of events, an annotation workspace and a training tab.

- **Model:** YOLOv8-based, 10 anomaly classes
- **Runtime:** TensorRT INT8 engine (13.3 MB)
- **Device:** NVIDIA Jetson Orin Nano (8 GB), JetPack R36.4.7, CUDA 12.6, TensorRT 10.3.0
- **Python:** 3.10, inside a venv at `CDAC-pothole/.venv`

### 10 anomaly classes

| id | label |
|---|---|
| 0 | pothole |
| 1 | crack_transverse |
| 2 | crack_longitudinal |
| 3 | rutting |
| 4 | debris |
| 5 | waterlogging |
| 6 | missing_lane_marking |
| 7 | edge_break |
| 8 | manhole |
| 9 | road_barrier |

### Key paths

| path | purpose |
|---|---|
| `inference/app.py` | Unified FastAPI dashboard (port 8000) |
| `inference/retrained/best_retrained_int8.engine` | **recommended** 10-class INT8 engine |
| `inference/best_int8.engine` | legacy 1-class INT8 engine |
| `inference/display_int8_live.py` | direct OpenCV display (no browser) |
| `inference/gnss.py` | NMEA GNSS reader |
| `inference/potholes.db` | SQLite event store |
| `inference/uploads/` | saved event snapshots |
| `inference/shell.html` | dashboard UI |
| `demo.mp4` (repo root's sibling) | test video |

---

## 2. One-time setup (Jetson)

```bash
cd "/home/edgeai/Documents/Srikaanth CDAC/CDAC-pothole"
python3 -m venv .venv
source .venv/bin/activate
pip install -r inference/requirements-jetson.txt
```

Verify the engine exists (it is git-ignored, must be present on disk):

```bash
ls -lh inference/retrained/best_retrained_int8.engine
# expected: ~13.3 MB
```

If it is missing, build it:

```bash
cd inference
../.venv/bin/python build_int8_engine.py \
  --onnx retrained/best_retrained.onnx \
  --calib-dir calib_images \
  --output retrained/best_retrained_int8.engine \
  --cache retrained/best_retrained_calibration.cache \
  --num-calib 500 --calib-batch-size 16
```

---

## 3. Start the complete dashboard (recommended)

```bash
cd "/home/edgeai/Documents/Srikaanth CDAC/CDAC-pothole"
.venv/bin/python inference/app.py \
  --host 0.0.0.0 --port 8000 \
  --engine inference/retrained/best_retrained_int8.engine \
  --video "/home/edgeai/Documents/Srikaanth CDAC/demo.mp4" \
  --loop
```

- `--loop` keeps the feed running after the 27 s clip ends so the dashboard stays live.
- Drop `--no-browser` is the default; the app opens Chromium automatically on `:1`. Add `--no-browser` for headless.
- Open `http://<jetson-ip>:8000/` in any browser.

### Optional flags

| flag | default | purpose |
|---|---|---|
| `--host` | `127.0.0.1` | bind address (`0.0.0.0` for LAN access) |
| `--port` | `8000` | HTTP port |
| `--engine` | `inference/best_int8.engine` | INT8 engine path (use the retrained one) |
| `--video` | `demo.mp4` | source video file |
| `--conf` | `0.15` | confidence threshold |
| `--nms` | `0.30` | NMS IoU threshold |
| `--gnss-port` | auto-detect | serial port of a GNSS receiver (`/dev/ttyUSB0`) |
| `--loop` | off | restart the clip when it ends |
| `--no-browser` | off | do not auto-open a browser |

### Environment-variable equivalents

`INT8_ENGINE`, `VIDEO_PATH`, `CONFIDENCE`, `NMS`, `GNSS_PORT`, `GNSS_BAUD`, `DB_PATH`, `UPLOADS_DIR`, `STATUS_PATH`, `ANNOTATION_DIR`.

---

## 4. What the dashboard shows

Four tabs in the top nav bar:

1. **Dashboard** — live MJPEG feed with per-class coloured boxes, FPS / detections / last-frame-ms stats, the GIS map of saved events, and the saved-events table with snapshot links.
2. **Annotate** — review & label frames, YOLO/JSON/COCO export, auto-save.
3. **Train** — retrain + build INT8 engine from the annotation workspace.
4. **Timeline** — pipeline stage status.

### Live stats endpoint

```bash
curl http://127.0.0.1:8000/video-feed-stats
# {"status":"running","fps":83.2,"detections":4,"frames":689,
#  "last_frame_ms":6.9,"active_boxes":7,"subscribers":1,"running":true}
```

### Other endpoints

| method | path | purpose |
|---|---|---|
| GET | `/` | dashboard (shell.html) |
| GET | `/health` | liveness + inference stats |
| GET | `/video-feed` | MJPEG stream |
| GET | `/video-feed-stats` | FPS / latency / subscriber count |
| GET | `/potholes` | all saved events (JSON) |
| GET | `/api/gnss` | GNSS state `{available, port, fix, latitude, longitude}` |
| POST | `/api/feed/start` | start / restart the feed |
| POST | `/api/feed/stop` | stop the feed |
| GET | `/api/pipeline/status` | pipeline stage status |
| GET | `/api/frames`, `/api/frames/{id}` | annotation frames |
| GET | `/api/export` | export annotations |
| GET | `/local-status` | local runner status |
| GET | `/uploads/...` | saved snapshots |

---

## 5. Direct-display mode (no browser)

For a pure OpenCV window on the Jetson desktop:

```bash
cd "/home/edgeai/Documents/Srikaanth CDAC/CDAC-pothole"
.venv/bin/python inference/display_int8_live.py \
  --source "/home/edgeai/Documents/Srikaanth CDAC/demo.mp4" \
  --engine inference/retrained/best_retrained_int8.engine \
  --conf 0.15 --nms 0.30 --limit 0
```

Press `q` in the window to quit. The window reports three separate numbers:
- **pipeline FPS** — frames actually inferred per second
- **pure TensorRT FPS** — `1000 / avg_infer_ms`
- **display FPS** — GUI-bound, what you actually see

---

## 6. GNSS setup (optional)

Plug in a USB/UART GNSS receiver (NMEA GGA/RMC). The port is auto-detected (`/dev/ttyUSB*`, `/dev/ttyACM*`), or set explicitly:

```bash
.venv/bin/python inference/app.py --port 8000 \
  --engine inference/retrained/best_retrained_int8.engine \
  --gnss-port /dev/ttyUSB0 --gnss-baud 9600 --loop
```

Without a fix, events are stored with `NULL` coordinates and are **not** plotted on the map (no Null-Island markers). Check state with `curl http://127.0.0.1:8000/api/gnss`.

---

## 7. Measured performance (Jetson Orin Nano)

| metric | retrained 10-class | original 1-class |
|---|---|---|
| inference-only FPS | 83.2 | 79.8 |
| web UI delivered FPS | **73.0** | **80.0** |
| server pipeline FPS | 80–87 | 89 |
| inference latency | 6.9–15.8 ms | 5.7–15.7 ms |
| engine size | 13.3 MB | 13.0 MB |

Stage costs per frame (10-class): decode 4.0 ms, preprocess 6.2 ms, TensorRT 15.0 ms, draw 0.6 ms, JPEG encode 8.9 ms.

The JPEG encode was the last bottleneck — moving it off the inference thread (it releases the GIL) took the pipeline from ~46 to ~73 FPS.

---

## 8. Stop / restart

```bash
# stop
pkill -f "inference/app.py"

# restart with the same command from section 3
```

Database backup is at `/tmp/kilo/potholes.db.bak` (taken before the column-order migration).

---

## 9. Troubleshooting

- **`Cuda Runtime (invalid resource handle)`** — the engine was created in a different thread. The app pushes the device context per call; if you add a new thread that touches CUDA, call `self.device_ctx.push()` / `.pop()` around it.
- **Qt `Timers cannot be stopped from another thread`** — only happens if you call `cv2.imshow`/`cv2.waitKey` from a non-main thread. Keep the GUI on the main thread.
- **No MJPEG frames / FPS stuck at 0** — the clip has finished. Use `--loop`, or POST `/api/feed/start`.
- **Map empty** — events have no GNSS fix (NULL coords). Attach a receiver or expect no markers.
- **`No images found`** (display script) — `--source` must be a video file or an image directory.
- **Engine missing** — engines are git-ignored; build with `build_int8_engine.py` (see section 2).

---

## 10. Files changed in the latest pass (not yet committed)

- `inference/app.py` — encoder thread, GNSS wiring, `--loop`, column-order fix, `/api/gnss`
- `inference/display_int8_live.py` — GUI on main thread, inference on worker, CUDA ctx push
- `inference/shell.html` — Null-Island map filter, GPS column
- `RETRAINED_MODEL_REPORT.txt`, `POC_GAP_ANALYSIS.txt`, `SESSION_SUMMARY_2026-10-21.txt` — updated numbers

Full state handover: `SESSION_SUMMARY_2026-10-21.txt` (read section 9 first).
