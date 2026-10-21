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
import asyncio
import atexit
import json
import logging
import os
import queue
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
    from annotate.env_setup import cuda_lib_env  # type: ignore
except Exception:  # running directly from inference/
    sys.path.insert(0, str(INFERENCE_DIR))
    from annotate.annotation_app import Workspace, get_workspace  # type: ignore
    from annotate.classes import CLASS_NAMES, SEVERITY_LEVELS  # type: ignore
    from annotate.schema import Detection  # type: ignore
    from annotate.env_setup import cuda_lib_env  # type: ignore


#: BGR colour per anomaly class, used for the live overlay.
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 255, 0),      # pothole
    1: (0, 200, 255),    # crack_transverse
    2: (255, 200, 0),    # crack_longitudinal
    3: (0, 165, 255),    # rutting
    4: (200, 200, 200),  # debris
    5: (255, 0, 255),    # waterlogging
    6: (255, 255, 0),    # missing_lane_marking
    7: (0, 255, 255),    # edge_break
    8: (128, 0, 255),    # manhole
    9: (255, 128, 0),    # road_barrier
}


class DetectionsBody(BaseModel):
    detections: list[Detection] = []


class MetaBody(BaseModel):
    telemetry: dict = {}
    environment: dict = {}


# ---------------------------------------------------------------------------
# Inference engine (INT8 TensorRT) - synchronous, runs in main thread
# where the pycuda.autoinit CUDA context is active
# ---------------------------------------------------------------------------

class TRTInference:
    """INT8 TensorRT engine wrapper.

    Uses page-locked (pinned) host buffers for roughly 2x faster H2D/D2H
    transfers, and exposes ``close()`` so device memory is released on
    shutdown - without that, a restart hits CUDA out-of-memory.

    The output tensor shape is read from the engine, so both the original
    single-class engine (1x5x8400) and the retrained 10-class engine
    (1x14x8400) are supported by the same code path.
    """

    def __init__(self, engine_path: Path):
        logger = trt.Logger(trt.Logger.WARNING)
        self.device_ctx = cuda.Device(0).make_context()
        self._closed = False
        try:
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.ctx = self.engine.create_execution_context()
            self.input_name = "images"
            self.output_name = "output0"
            # (1, 4 + num_classes, num_anchors)
            self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
            self.num_classes = self.output_shape[1] - 4
            self.d_input = cuda.mem_alloc(trt.volume((1, 3, IMAGE_SIZE, IMAGE_SIZE)) * 4)
            self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * 4)
            self.ctx.set_tensor_address(self.input_name, int(self.d_input))
            self.ctx.set_tensor_address(self.output_name, int(self.d_output))
            self.stream = cuda.Stream()
            # Pinned host buffers: the driver can DMA straight from/to them
            self.host_input = cuda.pagelocked_empty((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            self.host_output = cuda.pagelocked_empty(self.output_shape, dtype=np.float32)
        finally:
            self.device_ctx.pop()

    def close(self) -> None:
        """Free device memory and pop the CUDA context.

        Releases the GPU allocations so the process can be restarted without
        hitting CUDA out-of-memory when the engine is loaded again.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self.device_ctx.push()
        except Exception:  # noqa: BLE001 - context already gone
            return
        try:
            self.d_input.free()
            self.d_output.free()
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                self.device_ctx.pop()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def preprocess(frame):
        """CPU preprocess: BGR frame -> normalised NCHW float32 tensor."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.ascontiguousarray(np.expand_dims(img, axis=0))

    def infer_tensor(self, tensor, width, height, conf_thres, nms_thres):
        """Run TensorRT on a preprocessed tensor, return detections.

        ``width``/``height`` are the source frame dimensions, used to map
        the 640x640 model output back onto the original frame.

        The raw output is (4 + num_classes) x anchors. The best class per
        anchor is selected, then class-aware NMS suppresses overlapping
        boxes of the same class only.
        """
        self.device_ctx.push()
        try:
            np.copyto(self.host_input, tensor)
            cuda.memcpy_htod_async(self.d_input, self.host_input, self.stream)
            self.ctx.execute_async_v3(stream_handle=int(self.stream.handle))
            cuda.memcpy_dtoh_async(self.host_output, self.d_output, self.stream)
            self.stream.synchronize()
            sx = width / IMAGE_SIZE
            sy = height / IMAGE_SIZE
            pred = np.transpose(self.host_output[0], (1, 0))  # (anchors, 4 + nc)
            cls_scores = pred[:, 4:]
            class_ids = np.argmax(cls_scores, axis=1)
            scores = cls_scores[np.arange(len(class_ids)), class_ids]
            mask = scores >= conf_thres
            dets = []
            if np.any(mask):
                f = pred[mask]
                scores = scores[mask].astype(np.float32)
                class_ids = class_ids[mask]
                cxs, cys, ws, hs = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
                x1s = (cxs - ws / 2) * sx
                y1s = (cys - hs / 2) * sy
                x2s = (cxs + ws / 2) * sx
                y2s = (cys + hs / 2) * sy
                boxes = np.column_stack([x1s, y1s, x2s, y2s]).astype(np.float32)
                # Class-aware NMS: shifting each class into its own
                # coordinate region stops two different classes from
                # suppressing each other.
                offset = class_ids.astype(np.float32) * 7680.0
                nms_input = boxes.copy()
                nms_input[:, 0] += offset
                nms_input[:, 1] += offset
                idx = cv2.dnn.NMSBoxes(nms_input.tolist(), scores.tolist(), conf_thres, nms_thres)
                if len(idx):
                    for i in idx.flatten():
                        x1, y1, x2, y2 = boxes[i]
                        class_id = int(class_ids[i])
                        dets.append({
                            "bbox": [max(0, int(x1)), max(0, int(y1)),
                                     min(width - 1, int(x2)), min(height - 1, int(y2))],
                            "confidence": float(scores[i]),
                            "class_id": class_id,
                            "label": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                        })
            return dets
        finally:
            self.device_ctx.pop()

    def infer_detections(self, frame, conf_thres, nms_thres):
        """Preprocess + infer in one call (used by scripts and tests)."""
        h, w = frame.shape[:2]
        tensor = self.preprocess(frame)
        return self.infer_tensor(tensor, w, h, conf_thres, nms_thres)


BOX_TTL_FRAMES = 8  # how many frames a box stays visible after last detection


class BoxTracker:
    """Keeps detected boxes alive for N frames to prevent blinking/flickering.
    Matches new detections to existing boxes by IoU so boxes are refreshed
    in-place (no trails, no duplicates, no blinking)."""

    def __init__(self, ttl: int = BOX_TTL_FRAMES, iou_thresh: float = 0.3):
        self.ttl = ttl
        self.iou_thresh = iou_thresh
        self.boxes: list[dict] = []  # each: {bbox, confidence, ttl_left}

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        return inter / max(1.0, area_a + area_b - inter)

    def update(self, detections: list[dict]) -> list[dict]:
        # Decrement TTL on existing boxes
        alive = [b for b in self.boxes if b["ttl_left"] > 0]
        for b in alive:
            b["ttl_left"] -= 1

        # Match each new detection to an existing box by best IoU
        for d in detections:
            best_idx, best_iou = -1, 0.0
            for j, b in enumerate(alive):
                iou_v = self._iou(b["bbox"], d["bbox"])
                if iou_v > best_iou:
                    best_iou, best_idx = iou_v, j
            if best_idx >= 0 and best_iou >= self.iou_thresh:
                alive[best_idx]["bbox"] = d["bbox"]
                alive[best_idx]["confidence"] = d["confidence"]
                alive[best_idx]["ttl_left"] = self.ttl
                alive[best_idx]["class_id"] = d.get("class_id", 0)
                alive[best_idx]["label"] = d.get("label", "pothole")
            else:
                alive.append({
                    "bbox": d["bbox"],
                    "confidence": d["confidence"],
                    "ttl_left": self.ttl,
                    "class_id": d.get("class_id", 0),
                    "label": d.get("label", "pothole"),
                })

        self.boxes = alive
        return [b for b in self.boxes if b["ttl_left"] > 0]

    def clear(self):
        self.boxes = []


class InferenceLoop:
    """Three-thread inference pipeline.

    producer  : video decode + CPU preprocess -> pre_queue
    inference : pops preprocessed frames, runs TensorRT, tracks boxes,
                and JPEG-encodes only while a subscriber is watching
    db worker : batches pothole events into SQLite off the hot path

    Splitting the work this way keeps the GPU fed without the request
    thread paying for decode, preprocess, JPEG encoding and DB writes.
    """

    PRE_QUEUE_SIZE = 3
    EVENT_BATCH_SIZE = 25
    EVENT_FLUSH_SECONDS = 0.5

    def __init__(self, engine_path, video_path, conf, nms, every=1):
        self.engine_path = engine_path
        self.video_path = video_path
        self.conf = conf
        self.nms = nms
        self.every = every
        self.model = None
        self.cap = None
        self.tracker = BoxTracker(ttl=BOX_TTL_FRAMES)
        self.stats = {"status": "starting", "fps": 0.0, "detections": 0,
                      "frames": 0, "last_frame_ms": 0.0}
        self._t_start = time.perf_counter()
        self._n_frames = 0
        self._lock = threading.Lock()
        self._stop = False
        self._last_save = 0
        self._subscribers = 0
        self._latest_jpeg = None
        self._latest_seq = 0
        self._pre_queue = queue.Queue(maxsize=self.PRE_QUEUE_SIZE)
        self._event_queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._closed = False

    # -- lifecycle ---------------------------------------------------------
    @property
    def running(self):
        """True while at least one worker thread is alive."""
        return any(thread.is_alive() for thread in self._threads)

    def start(self):
        """Start the worker threads, restarting cleanly if already run before.

        Calling start() after stop() (or after the source video ran out) tears
        the previous run down first, so the GPU allocations and the capture
        are not leaked and the video restarts from the first frame.
        """
        if self.running:
            return
        self.close()  # release any previous run's GPU + capture
        self._closed = False
        self._stop = False
        self._pre_queue = queue.Queue(maxsize=self.PRE_QUEUE_SIZE)
        self._event_queue = queue.Queue()
        self._latest_jpeg = None
        self._latest_seq = 0
        self._last_save = 0
        self._t_start = time.perf_counter()
        self._n_frames = 0
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
        for target, name in ((self._producer_loop, "producer"),
                             (self._inference_loop, "inference"),
                             (self._db_loop, "db")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self):
        """Ask the worker threads to exit and wait for them to finish."""
        self._stop = True
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=3)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:  # noqa: BLE001
                pass
            self.cap = None
        if self.stats.get("status") == "running":
            self.stats["status"] = "stopped"

    def close(self):
        """Stop all threads and release the GPU. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._stop = True
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=3)
        self._threads.clear()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:  # noqa: BLE001
                pass
            self.cap = None
        if self.model is not None:
            self.model.close()
            self.model = None
        self.stats["status"] = "stopped"

    # -- subscribers -------------------------------------------------------
    def add_subscriber(self):
        with self._lock:
            self._subscribers += 1

    def remove_subscriber(self):
        with self._lock:
            self._subscribers = max(0, self._subscribers - 1)

    def subscriber_count(self):
        with self._lock:
            return self._subscribers

    def latest_frame(self):
        """Return (jpeg_bytes, sequence) for the most recent encoded frame."""
        with self._lock:
            return self._latest_jpeg, self._latest_seq

    # -- threads -----------------------------------------------------------
    def _producer_loop(self):
        """Decode + preprocess frames into the queue."""
        while not self._stop:
            try:
                ok, frame = self.cap.read()
            except Exception:  # noqa: BLE001
                break
            if not ok:
                if self.video_path:
                    self._stop = True  # plays once, no loop
                    with self._lock:
                        self.stats["status"] = "finished"
                break
            try:
                tensor = self.model.preprocess(frame)
                h, w = frame.shape[:2]
                self._pre_queue.put((tensor, frame, w, h), timeout=0.5)
            except queue.Full:
                continue  # inference is behind; drop this frame
            except Exception:  # noqa: BLE001
                logging.exception("Preprocess failed")
                break

    def _inference_loop(self):
        """Run TensorRT over preprocessed frames and publish JPEG output."""
        while not self._stop:
            try:
                tensor, frame, width, height = self._pre_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                t0 = time.perf_counter()
                dets = self.model.infer_tensor(tensor, width, height, self.conf, self.nms)
                infer_ms = (time.perf_counter() - t0) * 1000

                active_boxes = self.tracker.update(dets)

                # Draw once: the same rendering feeds the snapshot and the feed.
                for b in active_boxes:
                    x1, y1, x2, y2 = b["bbox"]
                    color = CLASS_COLORS.get(b.get("class_id", 0), (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{b.get('label', 'pothole')} {b['confidence']:.2f}",
                                (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                color, 2)

                # Queue events for the batched DB writer.
                now = time.perf_counter()
                if dets and now - self._last_save >= 1.0:
                    self._last_save = now
                    self._event_queue.put((frame, dets))

                # JPEG encode only while someone is watching the feed.
                jpeg = None
                if self.subscriber_count() > 0:
                    ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok2:
                        jpeg = buf.tobytes()

                with self._lock:
                    self.stats["frames"] += 1
                    self.stats["last_frame_ms"] = round(infer_ms, 1)
                    self.stats["detections"] = len(dets)
                    self.stats["active_boxes"] = len(active_boxes)
                    self.stats["status"] = "running"
                    self._n_frames += 1
                    elapsed = time.perf_counter() - self._t_start
                    if elapsed >= 1.0:
                        self.stats["fps"] = round(self._n_frames / elapsed, 1)
                        self._n_frames = 0
                        self._t_start = time.perf_counter()
                    if jpeg is not None:
                        self._latest_jpeg = jpeg
                        self._latest_seq += 1
            except Exception:  # noqa: BLE001
                logging.exception("Inference failed")
                continue

    def _db_loop(self):
        """Batch pothole events into SQLite with a single reused connection."""
        conn = None
        pending: list[tuple] = []
        last_flush = time.perf_counter()
        try:
            while not self._stop or pending or not self._event_queue.empty():
                try:
                    frame, dets = self._event_queue.get(timeout=0.2)
                    snap_url = self._write_snapshot(frame)
                    for d in dets:
                        pending.append((d.get("label", "pothole"), d["confidence"],
                                         0.0, 0.0, snap_url,
                                         datetime.now(timezone.utc).isoformat()))
                except queue.Empty:
                    pass
                except Exception:  # noqa: BLE001
                    logging.exception("Failed to record pothole event")
                now = time.perf_counter()
                if pending and (len(pending) >= self.EVENT_BATCH_SIZE
                                or now - last_flush >= self.EVENT_FLUSH_SECONDS):
                    if conn is None:
                        conn = sqlite3.connect(DB_PATH)
                    conn.executemany(
                        "INSERT INTO potholes (label, confidence, latitude, longitude, timestamp, image_path)"
                        " VALUES (?, ?, ?, ?, ?, ?)", pending)
                    conn.commit()
                    pending = []
                    last_flush = now
        finally:
            if conn is not None:
                try:
                    if pending:
                        conn.executemany(
                            "INSERT INTO potholes (label, confidence, latitude, longitude, timestamp, image_path)"
                            " VALUES (?, ?, ?, ?, ?, ?)", pending)
                        conn.commit()
                except Exception:  # noqa: BLE001
                    pass
                conn.close()

    def _write_snapshot(self, frame) -> str:
        """Persist an annotated snapshot and return its /uploads URL."""
        import uuid as _uuid
        name = f"pothole_{int(time.time()*1000)}_{_uuid.uuid4().hex[:8]}.jpg"
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(UPLOADS_DIR / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return f"/uploads/{name}"

    def get_stats(self):
        with self._lock:
            stats = dict(self.stats)
            stats["subscribers"] = self._subscribers
            stats["running"] = self.running
            return stats


_inference = None
_inference_lock = threading.Lock()


def get_inference() -> InferenceLoop:
    global _inference
    with _inference_lock:
        if _inference is None:
            engine = Path(os.getenv("INT8_ENGINE", str(INT8_ENGINE_DEFAULT)))
            video = os.getenv("VIDEO_PATH", "/home/edgeai/Documents/Srikaanth CDAC/demo.mp4")
            conf = float(os.getenv("CONFIDENCE", "0.15"))
            nms = float(os.getenv("NMS", "0.30"))
            _inference = InferenceLoop(engine, video, conf, nms)
            _inference.start()
        return _inference


def shutdown_inference():
    """Release GPU resources. Registered with atexit and FastAPI shutdown.

    Without this the CUDA allocations survive the process and the next
    start fails with "Cuda Runtime (out of memory)" while loading the
    engine.
    """
    global _inference
    with _inference_lock:
        loop = _inference
        _inference = None
    if loop is not None:
        logging.info("Shutting down inference pipeline")
        loop.close()


atexit.register(shutdown_inference)


@app.on_event("shutdown")
async def _release_inference():
    """Free the GPU when the ASGI server shuts down.

    The atexit hook alone is not enough for a Uvicorn reload or a clean
    SIGTERM: the process may keep the CUDA context alive and the next
    start fails with "Cuda Runtime (out of memory)".
    """
    shutdown_inference()


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


def save_pothole(label, confidence, lat, lng, image_path):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO potholes (label, confidence, latitude, longitude, timestamp, image_path) VALUES (?, ?, ?, ?, ?, ?)",
            (label, confidence, lat, lng, datetime.now(timezone.utc).isoformat(), image_path),
        )
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
    """Stream the live inference output as multipart MJPEG.

    The pipeline encodes JPEG only while at least one subscriber is
    connected, so this endpoint registers itself as a subscriber for the
    lifetime of the response and unregisters on disconnect.
    """
    inf = get_inference()

    async def gen():
        inf.add_subscriber()
        last_seq = -1
        last_yield = time.perf_counter()
        try:
            while True:
                jpeg, seq = inf.latest_frame()
                if jpeg is not None and seq != last_seq:
                    last_seq = seq
                    last_yield = time.perf_counter()
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                else:
                    await asyncio.sleep(0.01)
                    # End the stream once the pipeline is done and idle, so the
                    # browser does not wait forever on a video that has ended.
                    if not inf.running and (time.perf_counter() - last_yield) > 1.0:
                        break
        finally:
            inf.remove_subscriber()

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video-feed-stats")
async def video_feed_stats():
    return get_inference().get_stats()


@app.post("/api/feed/stop")
async def stop_feed():
    inf = get_inference()
    inf.stop()
    return {"ok": True, "status": inf.stats["status"]}


@app.post("/api/feed/start")
async def start_feed():
    inf = get_inference()
    inf.start()
    return {"ok": True, "status": inf.stats["status"]}


@app.get("/potholes")
async def get_potholes():
    return {"potholes": get_all_potholes()}


@app.delete("/api/potholes/clear")
async def clear_potholes():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM potholes")
            conn.commit()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        env=cuda_lib_env(),
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
    parser.add_argument("--video", default="/home/edgeai/Documents/Srikaanth CDAC/demo.mp4",
                        help="Video file (default: /home/edgeai/Documents/Srikaanth CDAC/demo.mp4)")
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
