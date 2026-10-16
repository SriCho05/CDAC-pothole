"""Unified CDAC Pothole Management App.

One command to run everything:

    python inference/app.py

Opens a single browser window with a timeline nav bar:

  Dashboard  (live INT8 inference + GIS map)
  Annotate   (review & label frames)
  Train      (retrain + build INT8 engine)
  Timeline   (pipeline stage status)

Uses the latest INT8 engine (inference/best_int8.engine) for live inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

INFERENCE_DIR = Path(__file__).resolve().parent
INT8_ENGINE_DEFAULT = INFERENCE_DIR / "best_int8.engine"
IMAGE_SIZE = 640
DB_PATH = Path(os.getenv("DB_PATH", INFERENCE_DIR / "potholes.db"))
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", INFERENCE_DIR / "uploads"))
STATUS_PATH = Path(os.getenv("STATUS_PATH", INFERENCE_DIR / "local_status.json"))
ANNOTATION_DIR = Path(os.getenv("ANNOTATION_DIR", INFERENCE_DIR / "annotations"))

app = FastAPI(title="CDAC Pothole Management App")

# ---------------------------------------------------------------------------
# Annotator workspace (shared with inference/annotate/annotation_app.py)
# ---------------------------------------------------------------------------
os.environ.setdefault("ANNOTATION_DIR", str(ANNOTATION_DIR))

try:  # packaged import
    from annotate.annotation_app import Workspace, get_workspace  # type: ignore
    from annotate.classes import CLASS_NAMES, SEVERITY_LEVELS  # type: ignore
    from annotate.schema import Detection  # type: ignore
except Exception:  # running directly from inference/
    sys.path.insert(0, str(INFERENCE_DIR))
    from annotate.annotation_app import Workspace, get_workspace  # type: ignore
    from annotate.classes import CLASS_NAMES, SEVERITY_LEVELS  # type: ignore
    from annotate.schema import Detection  # type: ignore


class DetectionsBody(BaseModel):
    detections: list[Detection] = []


class MetaBody(BaseModel):
    telemetry: dict = {}
    environment: dict = {}


# ---------------------------------------------------------------------------
# Inference engine (INT8 TensorRT) with background capture thread
# ---------------------------------------------------------------------------

class TRTInference:
    def __init__(self, engine_path: Path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.input_name = "images"
        self.output_name = "output0"
        self.d_input = cuda.mem_alloc(trt.volume((1, 3, IMAGE_SIZE, IMAGE_SIZE)) * 4)
        self.d_output = cuda.mem_alloc(trt.volume((1, 5, 8400)) * 4)
        self.ctx.set_tensor_address(self.input_name, int(self.d_input))
        self.ctx.set_tensor_address(self.output_name, int(self.d_output))
        self.stream = cuda.Stream()
        self.host_output = np.empty((1, 5, 8400), dtype=np.float32)

    def infer_detections(self, frame, conf_thres, nms_thres):
        h, w = frame.shape[:2]
        sx = w / IMAGE_SIZE
        sy = h / IMAGE_SIZE
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.ascontiguousarray(np.expand_dims(img, axis=0))
        cuda.memcpy_htod_async(self.d_input, img, self.stream)
        self.ctx.execute_async_v3(stream_handle=int(self.stream.handle))
        cuda.memcpy_dtoh_async(self.host_output, self.d_output, self.stream)
        self.stream.synchronize()
        pred = self.host_output[0]
        pred = np.transpose(pred, (1, 0))
        confs = pred[:, 4]
        mask = confs >= conf_thres
        dets = []
        if np.any(mask):
            f = pred[mask]
            scores = f[:, 4].astype(np.float32)
            cxs, cys, ws, hs = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
            x1s = (cxs - ws / 2) * sx
            y1s = (cys - hs / 2) * sy
            x2s = (cxs + ws / 2) * sx
            y2s = (cys + hs / 2) * sy
            boxes = np.column_stack([x1s, y1s, x2s, y2s]).astype(np.float32)
            idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, nms_thres)
            if len(idx):
                for i in idx.flatten():
                    x1, y1, x2, y2 = boxes[i]
                    dets.append({
                        "bbox": [max(0, int(x1)), max(0, int(y1)),
                                 min(w - 1, int(x2)), min(h - 1, int(y2))],
                        "confidence": float(scores[i]),
                    })
        return dets


class InferenceLoop:
    def __init__(self, engine_path, video_path, conf, nms, every=1):
        self.engine_path = engine_path
        self.video_path = video_path
        self.conf = conf
        self.nms = nms
        self.every = every
        self.model = None
        self.cap = None
        self.latest_jpeg = None
        self.stats = {"status": "starting", "fps": 0.0, "detections": 0,
                      "frames": 0, "last_frame_ms": 0.0}
        self._stop = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        try:
            self.model = TRTInference(self.engine_path)
        except Exception as e:
            logging.exception("Failed to load INT8 engine: %s", e)
            self.stats["status"] = "error"
            return
        source = self.video_path if self.video_path else 0
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.stats["status"] = "error"
            return
        self.stats["status"] = "running"
        frame_idx = 0
        t_start = time.perf_counter()
        n_frames = 0
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                if self.video_path:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            if frame_idx % max(1, self.every) != 0:
                frame_idx += 1
                continue
            frame_idx += 1
            t0 = time.perf_counter()
            dets = self.model.infer_detections(frame, self.conf, self.nms)
            infer_ms = (time.perf_counter() - t0) * 1000
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"pothole {d['confidence']:.2f}",
                            (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok2:
                with self._lock:
                    self.latest_jpeg = buf.tobytes()
            n_frames += 1
            elapsed = time.perf_counter() - t_start
            if elapsed >= 1.0:
                with self._lock:
                    self.stats["fps"] = round(n_frames / elapsed, 1)
                    self.stats["detections"] = len(dets)
                    self.stats["frames"] += n_frames
                    self.stats["last_frame_ms"] = round(infer_ms, 1)
                    self.stats["status"] = "running"
                n_frames = 0
                t_start = time.perf_counter()
        if self.cap:
            self.cap.release()

    def get_frame(self):
        with self._lock:
            return self.latest_jpeg

    def get_stats(self):
        with self._lock:
            return dict(self.stats)


_inference = None
_inference_lock = threading.Lock()


def get_inference() -> InferenceLoop:
    global _inference
    with _inference_lock:
        if _inference is None:
            engine = Path(os.getenv("INT8_ENGINE", str(INT8_ENGINE_DEFAULT)))
            video = os.getenv("VIDEO_PATH")  # None -> camera
            conf = float(os.getenv("CONFIDENCE", "0.15"))
            nms = float(os.getenv("NMS", "0.30"))
            _inference = InferenceLoop(engine, video, conf, nms)
            _inference.start()
        return _inference


# ---------------------------------------------------------------------------
# Database (potholes) - same schema as server.py
# ---------------------------------------------------------------------------

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS potholes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT, confidence REAL, latitude REAL, longitude REAL,
            timestamp TEXT, image_path TEXT)""")
        conn.commit()


def get_all_potholes():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM potholes ORDER BY timestamp DESC")
            return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []


init_db()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def shell():
    return FileResponse(INFERENCE_DIR / "shell.html")


@app.get("/video-feed")
async def video_feed():
    inf = get_inference()

    def gen():
        while True:
            jpeg = inf.get_frame()
            if jpeg is None:
                time.sleep(0.01)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(1.0 / 30.0)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video-feed-stats")
async def video_feed_stats():
    return get_inference().get_stats()


@app.get("/potholes")
async def get_potholes():
    return {"potholes": get_all_potholes()}


@app.get("/local-status")
async def local_status():
    try:
        return json.loads(STATUS_PATH.read_text()) if STATUS_PATH.exists() else {"status": "offline"}
    except Exception:
        return {"status": "offline"}


@app.get("/health")
async def health():
    return {"status": "ok", "inference": get_inference().get_stats()}


if UPLOADS_DIR.exists():
    app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


# ---------------------------------------------------------------------------
# Annotator API (served under /api so the annotator HTML's fetch() calls work)
# ---------------------------------------------------------------------------
@app.get("/annotate", response_class=HTMLResponse)
async def annotate_page():
    ui_path = Path(__file__).resolve().parent / "annotate" / "annotate.html"
    return FileResponse(ui_path)


@app.get("/api/classes")
async def api_classes():
    return {"classes": CLASS_NAMES, "severities": list(SEVERITY_LEVELS)}


@app.get("/api/frames")
async def api_frames(filter: str = "all"):
    return get_workspace().summaries(filter)


@app.get("/api/frames/{frame_id}")
async def api_get_frame(frame_id: str):
    record = get_workspace().get_record(frame_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown_frame")
    return record.to_dict()


@app.get("/api/frames/{frame_id}/image")
async def api_get_image(frame_id: str):
    workspace = get_workspace()
    entry = workspace.manifest.get(frame_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="unknown_frame")
    path = Path(entry["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="image_missing")
    return FileResponse(path)


@app.post("/api/frames/{frame_id}")
async def api_save_frame(frame_id: str, body: DetectionsBody):
    try:
        record = get_workspace().save_detections(frame_id, body.detections)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown_frame")
    return {"ok": True, "frame_id": frame_id, "detections": len(record.detections)}


@app.post("/api/frames/{frame_id}/meta")
async def api_save_meta(frame_id: str, body: MetaBody):
    try:
        get_workspace().save_meta(frame_id, body.telemetry, body.environment)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown_frame")
    return {"ok": True}


@app.delete("/api/frames/{frame_id}")
async def api_clear_frame(frame_id: str):
    try:
        get_workspace().clear_labels(frame_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown_frame")
    return {"ok": True}


@app.get("/api/export")
async def api_export():
    return get_workspace().export()


@app.get("/api/pipeline/status")
async def api_pipeline_status():
    return {"exported": get_workspace().export_done_flag.exists()}


@app.post("/api/pipeline/proceed")
async def api_pipeline_proceed():
    flag = get_workspace().export_done_flag
    if flag.exists():
        flag.unlink()
    return {"ok": True, "exported": False}


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
_train_proc = None
_train_log = []


@app.post("/api/train")
async def start_train():
    global _train_proc
    if _train_proc and _train_proc.poll() is None:
        return {"status": "already_running"}
    _train_log.clear()
    _train_proc = subprocess.Popen(
        [sys.executable, "-m", "annotate.train_pipeline",
         "--annotations", str(ANNOTATION_DIR),
         "--model", str(INFERENCE_DIR / "best.pt"),
         "--tensorrt", "--int8",
         "--calib-set", str(INFERENCE_DIR / "calib_images"),
         "--device", os.getenv("YOLO_DEVICE", "cuda:0")],
        cwd=str(INFERENCE_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    threading.Thread(target=_train_reader, daemon=True).start()
    return {"status": "started"}


def _train_reader():
    global _train_proc
    if _train_proc is None:
        return
    for line in _train_proc.stdout:
        _train_log.append(line.rstrip())
    _train_proc.wait()


@app.get("/api/train/status")
async def train_status():
    if _train_proc is None:
        return {"status": "idle", "log": _train_log}
    rc = _train_proc.poll()
    if rc is None:
        return {"status": "running", "log": _train_log[-50:]}
    return {"status": "done", "rc": rc, "log": _train_log[-50:]}


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------
@app.get("/api/timeline")
async def timeline():
    inf_stats = get_inference().get_stats()
    train_rc = _train_proc.poll() if _train_proc else None
    stages = [
        {"name": "Inference (INT8)", "status": inf_stats.get("status", "idle"),
         "detail": f"{inf_stats.get('fps', 0)} fps · {inf_stats.get('detections', 0)} dets"},
        {"name": "Annotate", "status": "ready",
         "detail": "Review & label frames at /annotate"},
        {"name": "Train", "status": "running" if train_rc is None and _train_proc else "idle",
         "detail": "Retrain + build INT8 engine"},
        {"name": "Deploy / Restart", "status": "ready",
         "detail": "Copy new engine to best_int8.engine & restart inference"},
    ]
    return {"stages": stages}


def main():
    parser = argparse.ArgumentParser(description="Unified CDAC Pothole Management App")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--engine", default=str(INT8_ENGINE_DEFAULT),
                        help="INT8 engine path for live inference")
    parser.add_argument("--video", default=None,
                        help="Video file (default: camera index 0)")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="INT8 confidence threshold")
    parser.add_argument("--nms", type=float, default=0.30,
                        help="NMS IoU threshold")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser window")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    os.environ["INT8_ENGINE"] = args.engine
    if args.video:
        os.environ["VIDEO_PATH"] = args.video
    os.environ["CONFIDENCE"] = str(args.conf)
    os.environ["NMS"] = str(args.nms)

    import uvicorn
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)

    if not args.no_browser:
        import webbrowser
        webbrowser.open(f"http://{args.host}:{args.port}/")

    server.run()


if __name__ == "__main__":
    main()
