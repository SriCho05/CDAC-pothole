"""Simple horizontal side-by-side: FP16 (old) left | INT8 (new) right.

Two panels only, IoU tracked boxes (no trails), TTL persistence (no blinking).
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

INFERENCE_DIR = Path(__file__).resolve().parent
ENGINE_FP16 = INFERENCE_DIR / "tests/artifacts/best_fp16.engine"
ENGINE_INT8 = INFERENCE_DIR / "best_int8.engine"
IMAGE_SIZE = 640
BOX_TTL_FRAMES = 8


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class BoxTracker:
    def __init__(self, ttl: int = BOX_TTL_FRAMES, iou_thresh: float = 0.3):
        self.ttl = ttl
        self.iou_thresh = iou_thresh
        self.boxes: list[dict] = []

    def update(self, detections: list[dict]) -> list[dict]:
        for b in self.boxes:
            b["ttl_left"] -= 1
        used_new = set()
        matched = []
        for b in self.boxes:
            best_iou = 0.0
            best_idx = -1
            for idx, d in enumerate(detections):
                if idx in used_new:
                    continue
                iou_v = iou(b["bbox"], d["bbox"])
                if iou_v > best_iou:
                    best_iou = iou_v
                    best_idx = idx
            if best_idx >= 0 and best_iou >= self.iou_thresh:
                used_new.add(best_idx)
                b["bbox"] = detections[best_idx]["bbox"]
                b["confidence"] = detections[best_idx]["confidence"]
                b["ttl_left"] = self.ttl
            matched.append(b)
        for idx, d in enumerate(detections):
            if idx not in used_new:
                matched.append({"bbox": d["bbox"], "confidence": d["confidence"], "ttl_left": self.ttl})
        self.boxes = [b for b in matched if b["ttl_left"] > 0]
        return self.boxes

    def clear(self):
        self.boxes = []


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
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, 0.45)
            if len(indices) > 0:
                for idx in indices.flatten():
                    x1, y1, x2, y2 = boxes[idx]
                    x1 = max(0, int(x1)); y1 = max(0, int(y1))
                    x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
                    conf = float(scores[idx])
                    detections.append({"bbox": [x1, y1, x2, y2], "confidence": conf})
        return detections

    def destroy(self):
        del self.d_input, self.d_output, self.stream, self.ctx, self.engine


def draw_boxes(frame, detections, color, label):
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        conf = d["confidence"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{label} {conf:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def main():
    parser = argparse.ArgumentParser(description="FP16 vs INT8 side-by-side, 2 panels")
    parser.add_argument("--source", required=True, help="Video file or image directory")
    parser.add_argument("--fp16-conf", type=float, default=0.25, help="FP16 confidence threshold")
    parser.add_argument("--int8-conf", type=float, default=0.15, help="INT8 confidence threshold")
    parser.add_argument("--nms", type=float, default=0.30, help="NMS IoU threshold")
    parser.add_argument("--limit", type=int, default=100, help="Max frames")
    parser.add_argument("--ttl", type=int, default=BOX_TTL_FRAMES, help="Box persistence frames")
    parser.add_argument("--iou-thresh", type=float, default=0.3, help="IoU match threshold")
    parser.add_argument("--output", default="fp16_vs_int8_2panel.mp4", help="Output MP4 path")
    args = parser.parse_args()

    src = Path(args.source)
    frames = []
    if src.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for path in sorted(src.iterdir()):
            if path.suffix.lower() in exts:
                img = cv2.imread(str(path))
                if img is not None:
                    frames.append(img)
                if len(frames) >= args.limit:
                    break
    elif src.is_file() and src.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}:
        cap = cv2.VideoCapture(str(src))
        while len(frames) < args.limit:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
    else:
        print("Source must be a video file or image directory.")
        return

    if not frames:
        print("No frames found.")
        return

    fp16 = TRTEngine(ENGINE_FP16, "FP16", (0, 255, 0))
    int8 = TRTEngine(ENGINE_INT8, "INT8", (0, 0, 255))
    tracker16 = BoxTracker(ttl=args.ttl, iou_thresh=args.iou_thresh)
    tracker8 = BoxTracker(ttl=args.ttl, iou_thresh=args.iou_thresh)

    h, w = frames[0].shape[:2]
    bar_h = 40
    panel_w = w if w % 2 == 0 else w + 1
    panel_h = h if h % 2 == 0 else h + 1
    out_w = panel_w * 2
    out_h = ((bar_h + panel_h) // 2) * 2

    writer = None
    for fourcc_name in ("mp4v", "avc1", "XVID", "DIVX"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        trial = cv2.VideoWriter(args.output, fourcc, 20.0, (out_w, out_h))
        if trial.isOpened():
            writer = trial
            print(f"Using codec: {fourcc_name}")
            break
    if writer is None:
        print("ERROR: Could not open VideoWriter.")
        return

    totals = {"fp16": 0, "int8": 0}
    start = time.perf_counter()
    try:
        for i, frame in enumerate(frames):
            fp16_dets = fp16.infer(frame, args.fp16_conf, args.nms)
            int8_dets = int8.infer(frame, args.int8_conf, args.nms)
            totals["fp16"] += len(fp16_dets)
            totals["int8"] += len(int8_dets)

            active16 = tracker16.update(fp16_dets)
            active8 = tracker8.update(int8_dets)

            fp16_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            fp16_panel[:h, :w] = frame
            fp16_panel = draw_boxes(fp16_panel, active16, (0, 255, 0), "FP16")

            int8_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            int8_panel[:h, :w] = frame
            int8_panel = draw_boxes(int8_panel, active8, (0, 0, 255), "INT8")

            bar = np.zeros((bar_h, out_w, 3), dtype=np.uint8)
            cv2.putText(bar, f"FP16@{args.fp16_conf}: {len(fp16_dets)}  |  INT8@{args.int8_conf}: {len(int8_dets)}  |  Frame {i+1}/{len(frames)}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            combined = np.vstack([bar, np.hstack([fp16_panel, int8_panel])])
            combined = np.ascontiguousarray(combined)
            writer.write(combined)

    finally:
        fp16.destroy()
        int8.destroy()
        writer.release()

    elapsed = time.perf_counter() - start
    print(f"Processed {len(frames)} frames in {elapsed:.1f}s ({len(frames)/elapsed:.1f} FPS)")
    print(f"FP16 total: {totals['fp16']}  INT8 total: {totals['int8']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
