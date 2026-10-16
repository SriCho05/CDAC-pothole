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
import time
from pathlib import Path

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

ENGINE_INT8 = Path(__file__).resolve().parent / "best_int8.engine"
IMAGE_SIZE = 640
BOX_TTL_FRAMES = 8  # how many frames a box stays visible after last detection


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
        for d in detections:
            self.boxes.append({
                "bbox": d["bbox"],
                "confidence": d["confidence"],
                "ttl_left": self.ttl,
            })

        return [b for b in self.boxes if b["ttl_left"] > 0]

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
        self.d_input = cuda.mem_alloc(trt.volume((1, 3, IMAGE_SIZE, IMAGE_SIZE)) * 4)
        self.d_output = cuda.mem_alloc(trt.volume((1, 5, 8400)) * 4)
        self.ctx.set_tensor_address(self.input_name, int(self.d_input))
        self.ctx.set_tensor_address(self.output_name, int(self.d_output))
        self.stream = cuda.Stream()
        self.host_output = np.empty((1, 5, 8400), dtype=np.float32)
        self.color = color

    def infer(self, frame, conf_thres, nms_thres):
        h, w = frame.shape[:2]
        scale_x = w / IMAGE_SIZE
        scale_y = h / IMAGE_SIZE
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
        out = frame.copy()
        detections = []
        if np.any(mask):
            filtered = pred[mask]
            scores = filtered[:, 4].astype(np.float32)
            cxs, cys, ws, hs = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
            x1s = (cxs - ws / 2) * scale_x
            y1s = (cys - hs / 2) * scale_y
            x2s = (cxs + ws / 2) * scale_x
            y2s = (cys + hs / 2) * scale_y
            boxes = np.column_stack([x1s, y1s, x2s, y2s]).astype(np.float32)
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, nms_thres)
            if len(indices) > 0:
                for idx in indices.flatten():
                    x1, y1, x2, y2 = boxes[idx]
                    x1 = max(0, int(x1)); y1 = max(0, int(y1))
                    x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
                    conf = float(scores[idx])
                    detections.append({"bbox": [x1, y1, x2, y2], "confidence": conf})
        return out, detections

    def destroy(self):
        del self.d_input, self.d_output, self.stream, self.ctx, self.engine


def main():
    parser = argparse.ArgumentParser(description="Live INT8 TensorRT inference display")
    parser.add_argument("--source", required=True, help="Video file or image directory")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.30, help="NMS IoU threshold")
    parser.add_argument("--limit", type=int, default=0, help="Max frames (0 = all)")
    parser.add_argument("--ttl", type=int, default=BOX_TTL_FRAMES, help="Box persistence in frames")
    args = parser.parse_args()

    src = Path(args.source)
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

    engine = TRTEngine(ENGINE_INT8, (0, 255, 0))
    tracker = BoxTracker(ttl=args.ttl)

    cap = None
    frame_idx = 0
    total_dets = 0
    timings = []
    start = time.perf_counter()

    if is_video:
        cap = cv2.VideoCapture(str(src))

    try:
        while True:
            if is_video:
                ok, frame = cap.read()
                if not ok:
                    break
                name = f"frame_{frame_idx:06d}"
            else:
                if frame_idx >= len(paths):
                    break
                frame = cv2.imread(str(paths[frame_idx]))
                if frame is None:
                    frame_idx += 1
                    continue
                name = paths[frame_idx].name

            t0 = time.perf_counter()
            out, detections = engine.infer(frame, args.conf, args.nms)
            infer_ms = (time.perf_counter() - t0) * 1000
            timings.append(infer_ms)

            # Update tracker with fresh detections; draw persistent boxes
            active_boxes = tracker.update(detections)
            total_dets += len(detections)
            for b in active_boxes:
                x1, y1, x2, y2 = b["bbox"]
                # Fade color based on remaining TTL
                alpha = b["ttl_left"] / args.ttl
                color = (0, int(255 * alpha), 0)
                thick = max(1, int(2 * alpha))
                cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

            # Overlay stats
            fps = 1000.0 / np.mean(timings) if timings else 0
            cv2.putText(out, f"INT8 | conf={args.conf} nms={args.nms}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(out, f"Live: {len(detections)}  Persistent: {len(active_boxes)}  "
                        f"Infer: {infer_ms:.1f}ms  FPS: {fps:.1f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(out, f"Frame: {name}  (press q to quit)",
                        (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("INT8 Live Inference", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            frame_idx += 1
            if args.limit and frame_idx >= args.limit:
                break
    finally:
        engine.destroy()
        if cap:
            cap.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    processed = frame_idx
    avg_ms = float(np.mean(timings)) if timings else 0
    print(f"Processed {processed} frames in {elapsed:.1f}s ({processed/elapsed:.1f} FPS wall)")
    print(f"Avg infer: {avg_ms:.1f}ms ({1000.0/avg_ms:.1f} infer FPS)" if avg_ms else "")
    print(f"Total detections: {total_dets} ({total_dets/max(processed,1):.2f}/frame)")


if __name__ == "__main__":
    main()
