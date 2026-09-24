import logging
import os
import time
import sqlite3
import json
import threading
import numpy as np
from datetime import datetime

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    trt = None
    cuda = None


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("yolo_pothole_detection")


API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is required for authentication.")


# --- YOLO Model for Video Feed ---
_model = None
_model_lock = threading.Lock()

def get_model():
    global _model
    with _model_lock:
        if _model is None:
            model_path = os.getenv("MODEL_PATH", os.path.join(INFERENCE_DIR, "best.pt"))
            device = os.getenv("YOLO_DEVICE", "cuda:0" if os.getenv("CUDA_VISIBLE_DEVICES") != "" else "cpu")
            if not YOLO:
                raise RuntimeError("Ultralytics not installed")
            _model = YOLO(model_path)
            _model.to(device)
        return _model


# --- TensorRT Model for Video Feed ---
_trt_model = None
_trt_model_lock = threading.Lock()

class TensorRTInference:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_shape = (1, 3, 640, 640)
        self.context.set_binding_shape(0, self.input_shape)
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            shape = self.context.get_binding_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
                self.bindings.append(int(device_mem))
            else:
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.outputs.append({"host": host_mem, "device": device_mem, "shape": shape})
                self.bindings.append(int(device_mem))

    def infer(self, frame):
        import numpy as np
        # Preprocess
        input_img = cv2.resize(frame, (640, 640))
        input_img = input_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        input_img = np.ascontiguousarray(input_img, dtype=np.float32) / 255.0
        input_img = np.expand_dims(input_img, 0)
        
        # Copy to GPU
        np.copyto(self.inputs[0]["host"], input_img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Copy outputs back
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()
        
        return self.outputs[0]["host"].reshape(self.outputs[0]["shape"])

def get_trt_model():
    global _trt_model
    with _trt_model_lock:
        if _trt_model is None:
            engine_path = os.getenv("TENSORRT_ENGINE", os.path.join(INFERENCE_DIR, "tests/artifacts/best_fp16.engine"))
            if not trt or not cuda:
                raise RuntimeError("TensorRT/pyCUDA not installed")
            _trt_model = TensorRTInference(engine_path)
        return _trt_model


# --- Database Setup ---
INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("DB_PATH", os.path.join(INFERENCE_DIR, "potholes.db"))
UPLOADS_DIR = os.getenv("UPLOADS_DIR", os.path.join(INFERENCE_DIR, "uploads"))

os.makedirs(UPLOADS_DIR, exist_ok=True)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS potholes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT,
                confidence REAL,
                latitude REAL,
                longitude REAL,
                timestamp TEXT,
                image_path TEXT
            )
        """)
        conn.commit()

init_db()

def save_pothole(label, confidence, lat, lng, image_path):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO potholes (label, confidence, latitude, longitude, timestamp, image_path) VALUES (?, ?, ?, ?, ?, ?)",
            (label, confidence, lat, lng, datetime.utcnow().isoformat(), image_path),
        )
        conn.commit()

def get_all_potholes():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM potholes ORDER BY timestamp DESC")
        return [dict(row) for row in cursor.fetchall()]


def require_api_key(x_api_key: str | None = Header(default=None)):
    if x_api_key != API_KEY:
        logger.warning("Authentication failed for request.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_api_key"
        )


app = FastAPI(title="YOLO Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

DASHBOARD_PATH = os.path.join(INFERENCE_DIR, "dashboard", "index.html")


@app.get("/", include_in_schema=False)
async def dashboard():
    """Serve the self-contained dashboard from the Jetson API process."""
    return FileResponse(DASHBOARD_PATH)


@app.get("/potholes", dependencies=[Depends(require_api_key)])
async def get_potholes():
    try:
        return {"potholes": get_all_potholes()}
    except Exception as exc:
        logger.exception("Failed to fetch potholes: %s", exc)
        raise HTTPException(status_code=500, detail="database_error")


@app.get("/health")
async def health():
    status_path = os.getenv("STATUS_PATH", os.path.join(os.path.dirname(__file__), "local_status.json"))
    local_status = None
    try:
        with open(status_path, encoding="utf-8") as status_file:
            local_status = json.load(status_file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {"status": "ok", "local_runner": local_status}


@app.get("/local-status")
async def local_status():
    """Return status written by the optional all-on-Jetson camera runner."""
    status_path = os.getenv("STATUS_PATH", os.path.join(os.path.dirname(__file__), "local_status.json"))
    try:
        with open(status_path, encoding="utf-8") as status_file:
            return json.load(status_file)
    except FileNotFoundError:
        return {"status": "not_running"}
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=503, detail="local_status_unavailable") from exc


# --- Live MJPEG Video Stream ---
_video_cap = None
_video_lock = threading.Lock()
_video_feed_stats = {"frames": 0, "fps": 0.0, "last_update": time.time()}

def get_video_capture():
    global _video_cap
    with _video_lock:
        if _video_cap is None or not _video_cap.isOpened():
            video_path = os.getenv("VIDEO_PATH", os.path.join(INFERENCE_DIR, "demo.mp4"))
            if not cv2:
                raise RuntimeError("OpenCV not installed")
            _video_cap = cv2.VideoCapture(video_path)
            if not _video_cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
        return _video_cap

def release_video_capture():
    global _video_cap
    with _video_lock:
        if _video_cap is not None:
            _video_cap.release()
            _video_cap = None

def mjpeg_generator():
    cap = get_video_capture()
    use_tensorrt = os.getenv("USE_TENSORRT", "true").lower() == "true"
    if use_tensorrt and trt and cuda:
        model = get_trt_model()
        is_tensorrt = True
    else:
        model = get_model()
        is_tensorrt = False
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_time = 1.0 / fps
    conf_thresh = float(os.getenv("CONFIDENCE", "0.40"))
    frame_count = 0
    start_time = time.time()
    
    while True:
        with _video_lock:
            if cap is None or not cap.isOpened():
                break
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Run inference
            if is_tensorrt:
                outputs = model.infer(frame)
                # Parse TensorRT output (assuming YOLO format: [batch, num_boxes, 85])
                # outputs shape: (1, 8400, 85) for YOLOv8
                detections = outputs[0]  # (8400, 85)
                for det in detections:
                    conf = det[4]
                    if conf < conf_thresh:
                        continue
                    cls_scores = det[5:]
                    cls_id = np.argmax(cls_scores)
                    cls_conf = cls_scores[cls_id]
                    if cls_conf < conf_thresh:
                        continue
                    x_center, y_center, w, h = det[0], det[1], det[2], det[3]
                    x1 = int((x_center - w/2) * frame.shape[1] / 640)
                    y1 = int((y_center - h/2) * frame.shape[0] / 640)
                    x2 = int((x_center + w/2) * frame.shape[1] / 640)
                    y2 = int((y_center + h/2) * frame.shape[0] / 640)
                    label = "pothole" if cls_id == 0 else str(cls_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label} {conf*cls_conf:.2f}", (x1, max(20, y1 - 8)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                results = model(frame, imgsz=640, conf=conf_thresh, verbose=False)[0]
                for box in results.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    if conf < conf_thresh:
                        continue
                    name = results.names[cls_id]
                    label = "pothole" if name == "0" or cls_id == 0 else name
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(20, y1 - 8)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            _video_feed_stats["fps"] = frame_count / elapsed
            _video_feed_stats["frames"] = frame_count
            _video_feed_stats["last_update"] = time.time()
            frame_count = 0
            start_time = time.time()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(frame_time)

@app.get("/video-feed")
async def video_feed():
    if not cv2:
        raise HTTPException(status_code=503, detail="OpenCV not available")
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/video-feed-stats")
async def video_feed_stats():
    return _video_feed_stats


@app.on_event("shutdown")
def shutdown_event():
    release_video_capture()
