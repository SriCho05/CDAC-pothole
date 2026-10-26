"""Live INT8 TensorRT inference display on Orin.

Shows INT8 detections in real-time with green bounding boxes, FPS counter,
and detection count overlay. Boxes persist across frames (temporal smoothing)
so they don't blink in/out as confidence hovers near the threshold.

Usage:
    python inference/display_int8_live.py \
        --source <video-file-or-image-dir> \
        --conf 0.15 \
        --nms 0.30 \
        --display
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

try:  # packaged import
    from annotate.classes import CLASS_NAMES  # type: ignore
except Exception:  # running directly from inference/
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from annotate.classes import CLASS_NAMES  # type: ignore

ENGINE_INT8 = Path(__file__).resolve().parent / "best_int8.engine"
IMAGE_SIZE = 640
BOX_TTL_FRAMES = 8  # how many frames a box stays visible after last detection

#: Sentinel pushed to the display slot when the source is exhausted.
DONE = object()


class FrameProducer:
    """Decodes and preprocesses frames on a background thread.

    Decode costs ~3.5 ms and preprocessing ~6.5 ms per frame. Done in the
    inference thread they are added to the GPU time; done here they overlap
    with it, which is the difference between the ~40 FPS this script used
    to reach and the ~80 FPS the headless benchmark gets.
    """

    def __init__(self, source, is_video, paths, image_size=IMAGE_SIZE, depth=3):
        self.source = source
        self.is_video = is_video
        self.paths = paths
        self.image_size = image_size
        self.queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def _loop(self):
        cap = cv2.VideoCapture(str(self.source)) if self.is_video else None
        index = 0
        try:
            while not self._stop.is_set():
                if self.is_video:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    name = f"frame_{index:06d}"
                else:
                    if index >= len(self.paths):
                        break
                    frame = cv2.imread(str(self.paths[index]))
                    if frame is None:
                        index += 1
                        continue
                    name = self.paths[index].name
                try:
                    tensor = TRTEngine.preprocess(frame)
                except Exception:  # noqa: BLE001
                    break
                try:
                    self.queue.put((tensor, frame, name), timeout=0.5)
                except queue.Full:
                    continue  # inference is behind; drop this frame
                index += 1
        finally:
            if cap is not None:
                cap.release()
            try:
                self.queue.put(None)
            except queue.Full:
                pass

    def read(self, timeout: float = 0.5):
        """Return the next (tensor, frame, name) or None when stopped."""
        while True:
            try:
                return self.queue.get(timeout=timeout)
            except queue.Empty:
                if self._stop.is_set():
                    return None

    def stop(self):
        self._stop.set()


class LatestFrame:
    """One-slot drop-oldest holder for the newest annotated frame.

    A slow display must never back-pressure the GPU, so the worker always
    overwrites the pending frame instead of waiting for the GUI to pick it
    up.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._item = None

    def put(self, item) -> None:
        with self._lock:
            self._item = item

    def take(self):
        """Return the pending item, or None if there is nothing new."""
        with self._lock:
            item = self._item
            self._item = None
            return item


class Display:
    """The OpenCV/Qt preview window, driven from the main thread.

    Qt requires that a window is created, painted and destroyed on the
    thread that owns the QApplication. Doing the GUI on a worker thread
    works, but the teardown prints
    ``QObject::killTimer: Timers cannot be stopped from another thread``
    and can abort at exit. So the GUI stays on the main thread and the
    inference pipeline moves to a worker instead - same throughput, clean
    shutdown.

    ``cv2.imshow`` + ``cv2.waitKey`` costs 5-9 ms per frame on this device:
    OpenCV here is built with the QT5 GUI backend and no QT OpenGL support,
    so every call does a software blit plus a Qt/X11 event round-trip to the
    desktop compositor. That is why the display rate is capped well below
    the inference rate.
    """

    WINDOW = "INT8 Live Inference"

    def __init__(self, window: str | None = None, scale: float = 1.0):
        self.window = window or self.WINDOW
        self.scale = scale
        self._open = False

    def open(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        self._open = True

    def show(self, frame) -> None:
        if self.scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale)
        cv2.imshow(self.window, frame)

    def poll(self) -> bool:
        """Pump the GUI event loop; True if the user asked to quit.

        waitKey is the expensive part, but it also services the Qt event
        loop, which is how the 'q' key is detected.
        """
        return (cv2.waitKey(1) & 0xFF) == ord("q")

    def close(self) -> None:
        if self._open:
            cv2.destroyWindow(self.window)
            self._open = False


class InferenceWorker(threading.Thread):
    """Runs decode -> TensorRT -> annotate on a background thread.

    The worker owns everything except the preview window: it pulls
    preprocessed frames from the producer, runs the engine, tracks and
    draws the boxes, optionally writes the annotated video, and publishes
    the finished frame to the display. The main thread only blits it.
    """

    def __init__(self, engine, producer, tracker, results, conf, nms,
                 writer=None, limit=0, ttl=BOX_TTL_FRAMES):
        super().__init__(name="inference", daemon=True)
        self.engine = engine
        self.producer = producer
        self.tracker = tracker
        self.results = results
        self.conf = conf
        self.nms = nms
        self.writer = writer
        self.limit = limit
        self.ttl = ttl
        self.timings: list[float] = []
        self.total_dets = 0
        self.frames = 0
        # Not ``self._stop``: threading.Thread already uses that name.
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        # Make the engine's CUDA context current in this thread.
        self.engine.cuda_ctx.push()
        try:
            self._run()
        finally:
            self.engine.cuda_ctx.pop()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                item = self.producer.read()
                if item is None:
                    break
                tensor, frame, name = item
                height, width = frame.shape[:2]

                t0 = time.perf_counter()
                out, detections = self.engine.infer_tensor(
                    tensor, width, height, self.conf, self.nms, frame)
                infer_ms = (time.perf_counter() - t0) * 1000
                self.timings.append(infer_ms)

                # Update tracker with fresh detections; draw persistent boxes
                active_boxes = self.tracker.update(detections)
                self.total_dets += len(detections)
                for b in active_boxes:
                    x1, y1, x2, y2 = b["bbox"]
                    # Fade color based on remaining TTL
                    alpha = b["ttl_left"] / max(self.ttl, 1)
                    color = (0, int(255 * alpha), 0)
                    thick = max(1, int(2 * alpha))
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
                    cv2.putText(out, f"{b.get('label', 'pothole')} {b['confidence']:.2f}",
                                (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 1)

                # Overlay stats
                fps = 1000.0 / np.mean(self.timings) if self.timings else 0
                cv2.putText(out, f"INT8 | conf={self.conf} nms={self.nms}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(out, f"Live: {len(detections)}  Persistent: {len(active_boxes)}  "
                            f"Infer: {infer_ms:.1f}ms  FPS: {fps:.1f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(out, f"Frame: {name}  (press q to quit)",
                            (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1)

                if self.writer is not None:
                    self.writer.write(out)
                self.results.put(out)

                self.frames += 1
                if self.limit and self.frames >= self.limit:
                    break
        finally:
            # Sentinel: tells the display loop the source ran out.
            self.results.put(DONE)


class BoxTracker:
    """Keeps detected boxes alive for N frames to prevent blinking."""

    def __init__(self, ttl: int = BOX_TTL_FRAMES):
        self.ttl = ttl
        self.boxes: list[dict] = []  # each: {bbox, conf, ttl_left}

    def update(self, detections: list[dict]) -> list[dict]:
        # Decrement TTL on existing boxes and drop expired ones
        alive = [b for b in self.boxes if b["ttl_left"] > 0]
        for b in alive:
            b["ttl_left"] -= 1

        # Add/refresh new detections (simple: append new, don't try to match IOU)
        alive.extend({
            "bbox": d["bbox"],
            "confidence": d["confidence"],
            "ttl_left": self.ttl,
            "class_id": d.get("class_id", 0),
            "label": d.get("label", "pothole"),
        } for d in detections)

        # Store the pruned list back. Without this the list grows for the
        # whole run: every frame appended to self.boxes while the expired
        # entries were only ever filtered out of the return value.
        self.boxes = alive
        return alive

    def clear(self):
        self.boxes = []


class TRTEngine:
    def __init__(self, engine_path, color):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.input_name = "images"
        self.output_name = "output0"
        # The output shape comes from the engine: (1, 5, 8400) for the
        # 1-class engine, (1, 14, 8400) for the retrained 10-class engine.
        self.output_shape = tuple(self.ctx.get_tensor_shape(self.output_name))
        self.num_classes = self.output_shape[1] - 4
        self.d_input = cuda.mem_alloc(trt.volume((1, 3, IMAGE_SIZE, IMAGE_SIZE)) * 4)
        self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * 4)
        self.ctx.set_tensor_address(self.input_name, int(self.d_input))
        self.ctx.set_tensor_address(self.output_name, int(self.d_output))
        self.stream = cuda.Stream()
        self.host_output = np.empty(self.output_shape, dtype=np.float32)
        # Pinned staging buffer for the input tensor.
        self.pinned_input = cuda.pagelocked_empty((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        # The CUDA context is per-thread: a worker thread gets a fresh one
        # unless the context that owns the device pointers is pushed onto
        # it. The engine is built in the main thread, so the inference
        # worker has to push this exact context to use it.
        self.cuda_ctx = cuda.Context.get_current()
        self.color = color

    @staticmethod
    def preprocess(frame):
        """BGR frame -> normalised NCHW float32 tensor."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.ascontiguousarray(np.expand_dims(img, axis=0))

    def infer_tensor(self, tensor, width, height, conf_thres, nms_thres, frame):
        """Run TensorRT on a preprocessed tensor and return (frame, detections).

        The tensor is staged through a pinned host buffer, which lets the
        driver DMA straight from it instead of copying from pageable memory.
        """
        scale_x = width / IMAGE_SIZE
        scale_y = height / IMAGE_SIZE
        np.copyto(self.pinned_input, tensor)
        cuda.memcpy_htod_async(self.d_input, self.pinned_input, self.stream)
        self.ctx.execute_async_v3(stream_handle=int(self.stream.handle))
        cuda.memcpy_dtoh_async(self.host_output, self.d_output, self.stream)
        self.stream.synchronize()

        pred = np.transpose(self.host_output[0], (1, 0))
        cls_scores = pred[:, 4:]
        if self.num_classes <= 1:
            class_ids = np.zeros(len(pred), dtype=np.int32)
            scores_all = cls_scores[:, 0]
        else:
            class_ids = np.argmax(cls_scores, axis=1)
            scores_all = cls_scores[np.arange(len(class_ids)), class_ids]
        mask = scores_all >= conf_thres
        out = frame.copy()
        detections = []
        if np.any(mask):
            filtered = pred[mask]
            scores = scores_all[mask].astype(np.float32)
            class_ids = class_ids[mask]
            cxs, cys, ws, hs = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
            x1s = (cxs - ws / 2) * scale_x
            y1s = (cys - hs / 2) * scale_y
            x2s = (cxs + ws / 2) * scale_x
            y2s = (cys + hs / 2) * scale_y
            boxes = np.column_stack([x1s, y1s, x2s, y2s]).astype(np.float32)
            # Class-aware NMS: each class gets its own coordinate region.
            offset = class_ids.astype(np.float32) * 7680.0
            nms_input = boxes.copy()
            nms_input[:, 0] += offset
            nms_input[:, 1] += offset
            indices = cv2.dnn.NMSBoxes(nms_input.tolist(), scores.tolist(), conf_thres, nms_thres)
            if len(indices) > 0:
                for idx in indices.flatten():
                    x1, y1, x2, y2 = boxes[idx]
                    x1 = max(0, int(x1)); y1 = max(0, int(y1))
                    x2 = min(width - 1, int(x2)); y2 = min(height - 1, int(y2))
                    conf = float(scores[idx])
                    class_id = int(class_ids[idx])
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_id": class_id,
                        "label": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    })
        return out, detections

    def infer(self, frame, conf_thres, nms_thres):
        """Preprocess + infer in one call."""
        h, w = frame.shape[:2]
        tensor = self.preprocess(frame)
        return self.infer_tensor(tensor, w, h, conf_thres, nms_thres, frame)

    def destroy(self):
        del self.d_input, self.d_output, self.pinned_input, self.stream, self.ctx, self.engine


def main():
    parser = argparse.ArgumentParser(description="Live INT8 TensorRT inference display")
    parser.add_argument("--source", required=True, help="Video file or image directory")
    parser.add_argument("--engine", default=str(ENGINE_INT8),
                        help="TensorRT engine to run (1-class or 10-class)")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.30, help="NMS IoU threshold")
    parser.add_argument("--limit", type=int, default=0, help="Max frames (0 = all)")
    parser.add_argument("--ttl", type=int, default=BOX_TTL_FRAMES, help="Box persistence in frames")
    parser.add_argument("--save-video", default=None, help="Also write the display to a video file")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale the preview window (1.0 = full size)")
    args = parser.parse_args()

    src = Path(args.source)
    paths: list[Path] = []
    if src.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted([p for p in src.iterdir() if p.suffix.lower() in exts])
        if not paths:
            print("No images found.")
            return
        is_video = False
    elif src.is_file() and src.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}:
        is_video = True
    else:
        print("Source must be a video file or image directory.")
        return

    engine = TRTEngine(args.engine, (0, 255, 0))
    tracker = BoxTracker(ttl=args.ttl)
    print(f"Engine: {args.engine}  classes: {engine.num_classes}")

    total_dets = 0
    timings: list[float] = []
    start = time.perf_counter()
    writer = None

    if args.save_video:
        src_cap = cv2.VideoCapture(str(src) if is_video else str(paths[0]))
        w = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = src_cap.get(cv2.CAP_PROP_FPS) or 25.0
        src_cap.release()
        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"Saving display to {args.save_video}")

    producer = FrameProducer(src, is_video, paths)
    results = LatestFrame()
    worker = InferenceWorker(engine, producer, tracker, results,
                             args.conf, args.nms, writer=writer,
                             limit=args.limit, ttl=args.ttl)
    display = Display(scale=args.scale)

    displayed = 0
    try:
        display.open()
        producer.start()
        worker.start()

        while True:
            out = results.take()
            if out is None:
                # Nothing new yet: yield the CPU instead of busy-waiting.
                time.sleep(0.001)
                if not worker.is_alive():
                    break
                continue
            if out is DONE:
                break
            display.show(out)
            displayed += 1
            if display.poll():
                print("Quit requested from the display window")
                break
    finally:
        worker.stop()
        producer.stop()
        worker.join(timeout=3)
        display.close()
        engine.destroy()
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - start
    processed = worker.frames
    timings = worker.timings
    total_dets = worker.total_dets
    avg_ms = float(np.mean(timings)) if timings else 0
    infer_fps = 1000.0 / avg_ms if avg_ms else 0
    print(f"Inference: {processed} frames in {elapsed:.1f}s "
          f"({processed / max(elapsed, 1e-6):.1f} FPS pipeline)")
    print(f"Avg infer: {avg_ms:.1f}ms ({infer_fps:.1f} FPS pure TensorRT)")
    print(f"Display: {displayed} frames ({displayed / max(elapsed, 1e-6):.1f} FPS, "
          f"GUI-bound)")
    print(f"Total detections: {total_dets} ({total_dets / max(processed, 1):.2f}/frame)")


if __name__ == "__main__":
    main()
