"""Side-by-side FP16 vs INT8 TensorRT display on Orin.

Shows both engines running on the same frames with bounding boxes drawn,
stacked vertically for direct visual comparison of what each detects.

Usage:
    python inference/display_fp16_vs_int8.py \
        --source <video-or-image-dir> \
        [--conf 0.25] [--limit 200] [--display]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

INFERENCE_DIR = Path(__file__).resolve().parent
ENGINE_FP16 = INFERENCE_DIR / "tests/artifacts/best_fp16.engine"
ENGINE_INT8 = INFERENCE_DIR / "best_int8.engine"
IMAGE_SIZE = 640


class TRTEngine:
    def __init__(self, engine_path: str | Path, label: str, color):
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
        self.label = label
        self.color = color

    def infer(self, frame: np.ndarray, conf_thres: float, nms_thres: float):
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
        pred = np.transpose(pred, (1, 0))  # (8400, 5)
        confs = pred[:, 4]
        mask = confs >= conf_thres
        if not np.any(mask):
            return frame.copy(), []
        filtered = pred[mask]
        scores = filtered[:, 4].astype(np.float32)
        cxs, cys, ws, hs = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
        x1s = (cxs - ws / 2) * scale_x
        y1s = (cys - hs / 2) * scale_y
        x2s = (cxs + ws / 2) * scale_x
        y2s = (cys + hs / 2) * scale_y
        boxes = np.column_stack([x1s, y1s, x2s, y2s]).astype(np.float32)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, nms_thres)

        out = frame.copy()
        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                x1, y1, x2, y2 = boxes[idx]
                x1 = max(0, int(x1)); y1 = max(0, int(y1))
                x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
                conf = float(scores[idx])
                cv2.rectangle(out, (x1, y1), (x2, y2), self.color, 2)
                cv2.putText(out, f"{self.label} {conf:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
                detections.append({"bbox": [x1, y1, x2, y2], "confidence": conf})
        return out, detections

    def destroy(self):
        del self.d_input, self.d_output, self.stream, self.ctx, self.engine


def collect_frames(source: str, limit: int) -> list:
    frames = []
    p = Path(source)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for path in sorted(p.iterdir()):
            if path.suffix.lower() in exts:
                img = cv2.imread(str(path))
                if img is not None:
                    frames.append((path.name, img))
                if len(frames) >= limit:
                    break
    elif p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}:
        cap = cv2.VideoCapture(str(p))
        idx = 0
        while len(frames) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append((f"frame_{idx:06d}", frame))
            idx += 1
        cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description="Side-by-side FP16 vs INT8 TensorRT display")
    parser.add_argument("--source", required=True, help="Video file or image directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--limit", type=int, default=200, help="Max frames")
    parser.add_argument("--display", action="store_true", default=True, help="Show windows")
    parser.add_argument("--output", default=None, help="Save combined output video")
    args = parser.parse_args()

    frames = collect_frames(args.source, args.limit)
    if not frames:
        print("No frames found.")
        return

    fp16 = TRTEngine(ENGINE_FP16, "FP16", (0, 255, 0))   # green
    int8 = TRTEngine(ENGINE_INT8, "INT8", (0, 0, 255))   # red

    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        sample = frames[0][1]
        h, w = sample.shape[:2]
        out_writer = cv2.VideoWriter(args.output, fourcc, 20.0, (w, h * 2 + 60))

    totals = {"fp16": 0, "int8": 0}
    start = time.perf_counter()

    try:
        for i, (name, frame) in enumerate(frames):
            fp16_img, fp16_dets = fp16.infer(frame, args.conf, args.nms)
            int8_img, int8_dets = int8.infer(frame, args.conf, args.nms)
            totals["fp16"] += len(fp16_dets)
            totals["int8"] += len(int8_dets)

            # Header bar with counts
            bar = np.zeros((40, frame.shape[1], 3), dtype=np.uint8)
            cv2.putText(bar, f"FP16: {len(fp16_dets)} dets   |   INT8: {len(int8_dets)} dets   |   Frame {i+1}/{len(frames)}: {name}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            combined = np.vstack([bar, fp16_img, int8_img])

            if out_writer:
                out_writer.write(combined)

            if args.display:
                cv2.imshow("FP16 vs INT8 TensorRT", combined)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
    finally:
        fp16.destroy()
        int8.destroy()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    processed = i + 1
    print(f"Processed {processed} frames in {elapsed:.1f}s ({processed/elapsed:.1f} FPS)")
    print(f"FP16 total detections: {totals['fp16']} ({totals['fp16']/processed:.2f}/frame)")
    print(f"INT8 total detections: {totals['int8']} ({totals['int8']/processed:.2f}/frame)")
    if args.output:
        print(f"Saved combined video: {args.output}")


if __name__ == "__main__":
    main()
