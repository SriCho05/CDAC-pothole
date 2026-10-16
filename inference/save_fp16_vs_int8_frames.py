"""Save side-by-side FP16 vs INT8 comparison frames as viewable images."""
from __future__ import annotations

import argparse
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
    def __init__(self, engine_path, label, color):
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
                    cv2.rectangle(out, (x1, y1), (x2, y2), self.color, 2)
                    cv2.putText(out, f"{self.label} {conf:.2f}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
                    detections.append({"bbox": [x1, y1, x2, y2], "confidence": conf})
        return out, detections

    def destroy(self):
        del self.d_input, self.d_output, self.stream, self.ctx, self.engine


def main():
    parser = argparse.ArgumentParser(description="Save FP16 vs INT8 side-by-side frame images")
    parser.add_argument("--source", required=True, help="Video file or image directory")
    parser.add_argument("--out", default="fp16_vs_int8_frames", help="Output directory for comparison images")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--limit", type=int, default=20, help="Max frames to save")
    parser.add_argument("--every", type=int, default=5, help="Save every Nth frame")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.source)
    frames = []
    if src.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for path in sorted(src.iterdir()):
            if path.suffix.lower() in exts:
                img = cv2.imread(str(path))
                if img is not None:
                    frames.append((path.name, img))
    elif src.is_file() and src.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}:
        cap = cv2.VideoCapture(str(src))
        idx = 0
        while len(frames) < args.limit * args.every:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append((f"frame_{idx:06d}", frame))
            idx += 1
        cap.release()

    fp16 = TRTEngine(ENGINE_FP16, "FP16", (0, 255, 0))
    int8 = TRTEngine(ENGINE_INT8, "INT8", (0, 0, 255))

    saved = 0
    totals = {"fp16": 0, "int8": 0}
    try:
        for i, (name, frame) in enumerate(frames):
            if i % args.every != 0:
                continue
            if saved >= args.limit:
                break
            fp16_img, fp16_dets = fp16.infer(frame, args.conf, args.nms)
            int8_img, int8_dets = int8.infer(frame, args.conf, args.nms)
            totals["fp16"] += len(fp16_dets)
            totals["int8"] += len(int8_dets)

            h, w = frame.shape[:2]
            bar = np.zeros((40, w, 3), dtype=np.uint8)
            cv2.putText(bar, f"FP16: {len(fp16_dets)}  |  INT8: {len(int8_dets)}  |  {name}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            combined = np.vstack([bar, fp16_img, int8_img])

            out_path = out_dir / f"{saved:03d}_{name}"
            cv2.imwrite(str(out_path), combined)
            saved += 1
    finally:
        fp16.destroy()
        int8.destroy()

    print(f"Saved {saved} side-by-side images to {out_dir.resolve()}")
    print(f"FP16 total: {totals['fp16']}  INT8 total: {totals['int8']}")


if __name__ == "__main__":
    main()
