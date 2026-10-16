"""Compare FP16 vs INT8 at multiple confidence thresholds on the same frames.

Generates side-by-side videos per threshold + JSON report of detection counts
and FPS for each config.
"""
from __future__ import annotations

import argparse
import json
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
NMS_THRES = 0.45


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


def collect_frames(source, limit):
    frames = []
    p = Path(source)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for path in sorted(p.iterdir()):
            if path.suffix.lower() in exts:
                img = cv2.imread(str(path))
                if img is not None:
                    frames.append(img)
                if len(frames) >= limit:
                    break
    elif p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}:
        cap = cv2.VideoCapture(str(p))
        while len(frames) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
    return frames


def even_dims(arr):
    h, w = arr.shape[:2]
    if h % 2 == 1:
        arr = np.vstack([arr, np.zeros((1, arr.shape[1], 3), dtype=np.uint8)])
    if w % 2 == 1:
        arr = np.hstack([arr, np.zeros((arr.shape[0], 1, 3), dtype=np.uint8)])
    return arr


def main():
    parser = argparse.ArgumentParser(description="FP16 vs INT8 multi-threshold comparison")
    parser.add_argument("--source", required=True, help="Video file or image directory")
    parser.add_argument("--limit", type=int, default=100, help="Max frames")
    parser.add_argument("--outdir", default="threshold_comparison", help="Output directory")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--thresholds", default="0.25,0.20,0.15,0.10", help="INT8 conf thresholds to test")
    parser.add_argument("--fp16-threshold", type=float, default=0.25, help="FP16 conf threshold")
    parser.add_argument("--display", action="store_true", help="Show live comparison window while running")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = collect_frames(args.source, args.limit)
    if not frames:
        print("No frames found.")
        return

    h, w = frames[0].shape[:2]
    bar_h = 40
    out_w = w if w % 2 == 0 else w + 1
    out_h = ((bar_h + h * 2) // 2) * 2

    fp16 = TRTEngine(ENGINE_FP16, "FP16", (0, 255, 0))
    int8 = TRTEngine(ENGINE_INT8, "INT8", (0, 0, 255))

    thresholds = [float(t) for t in args.thresholds.split(",")]

    report = {
        "source": str(args.source),
        "frames_tested": len(frames),
        "image_size": IMAGE_SIZE,
        "nms_thres": args.nms,
        "fp16_threshold": args.fp16_threshold,
        "int8_thresholds": thresholds,
        "results": [],
    }

    # FP16 baseline once
    fp16_results = []
    fp16_total = 0
    fp16_start = time.perf_counter()
    for frame in frames:
        fp16_img, fp16_dets = fp16.infer(frame, args.fp16_threshold, args.nms)
        fp16_total += len(fp16_dets)
        fp16_results.append((fp16_img, len(fp16_dets)))
    fp16_elapsed = time.perf_counter() - fp16_start
    fp16_fps = len(frames) / fp16_elapsed

    for th in thresholds:
        int8_total = 0
        int8_start = time.perf_counter()
        writers = {}
        combined_paths = []

        # Save one comparison video per threshold: FP16 vs INT8@th
        vid_name = f"fp16_vs_int8_conf{th}.mp4"
        vid_path = outdir / vid_name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(vid_path), fourcc, 20.0, (out_w, out_h))
        if not writer.isOpened():
            writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"avc1"), 20.0, (out_w, out_h))

        for i, frame in enumerate(frames):
            fp16_img, fp16_count = fp16_results[i]
            int8_img, int8_dets = int8.infer(frame, th, args.nms)
            int8_total += len(int8_dets)

            bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
            cv2.putText(bar, f"FP16@{args.fp16_threshold}: {fp16_count}  |  INT8@{th}: {len(int8_dets)}  |  Frame {i+1}/{len(frames)}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            combined = np.vstack([bar, fp16_img, int8_img])
            combined = even_dims(combined)
            writer.write(combined)

            if args.display:
                cv2.imshow(f"FP16 vs INT8@{th}", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        writer.release()
        int8_elapsed = time.perf_counter() - int8_start
        int8_fps = len(frames) / int8_elapsed

        recovered = int8_total / max(fp16_total, 1)
        lost = fp16_total - int8_total

        result = {
            "int8_threshold": th,
            "fp16_detections": fp16_total,
            "int8_detections": int8_total,
            "detections_lost": lost,
            "recovery_ratio": round(recovered, 4),
            "fp16_fps": round(fp16_fps, 2),
            "int8_fps": round(int8_fps, 2),
            "speedup": round(int8_fps / max(fp16_fps, 1e-9), 3),
            "video": str(vid_path),
        }
        report["results"].append(result)
        print(f"INT8 conf={th}: {int8_total} dets vs FP16 {fp16_total} "
              f"(recovery {recovered*100:.1f}%) | INT8 {int8_fps:.1f} FPS vs FP16 {fp16_fps:.1f} FPS")

    fp16.destroy()
    int8.destroy()

    report_path = outdir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path.resolve()}")
    print("Videos saved in:", outdir.resolve())


if __name__ == "__main__":
    main()
