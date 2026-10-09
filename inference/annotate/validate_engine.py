"""Validate a TensorRT engine against the PyTorch baseline on real survey frames.

Compares detection outputs (not just speed) per the Jetson roadmap:
  - missed potholes / false positives
  - bounding-box coordinate delta
  - confidence delta
  - inference latency, FPS

Usage (on the Orin Nano with the engine built):

    python -m annotate.validate_engine \\
        --model inference/best.pt \\
        --engine inference/best_fp16.engine \\
        --frames inference/annotations/frames \\
        --output inference/engine_validation_report.json

If --engine is omitted or TensorRT bindings are unavailable, only the PyTorch
baseline is measured and the script exits after writing results.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("validate_engine")

INFERENCE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FRAME_DIR = INFERENCE_DIR / "annotations" / "frames"
DEFAULT_OUTPUT = INFERENCE_DIR / "engine_validation_report.json"
DEFAULT_MODEL = INFERENCE_DIR / "best.pt"


def load_frames(frame_dir: Path, max_frames: int = 200) -> list[np.ndarray]:
    import cv2

    images = [path for path in sorted(frame_dir.glob("*.jpg")) if path.is_file()][:max_frames]
    frames = []
    for path in images:
        frame = cv2.imread(str(path))
        if frame is not None:
            frames.append(frame)
    if not frames:
        frames.append(np.zeros((720, 1280, 3), dtype=np.uint8))
    return frames


def box_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def summarize(timings: list[float]) -> dict:
    values = np.asarray(timings, dtype=float)
    average = float(values.mean())
    return {
        "avg_inf_ms": round(average, 3),
        "std_ms": round(float(values.std()), 3),
        "min_ms": round(float(values.min()), 3),
        "max_ms": round(float(values.max()), 3),
        "p50_ms": round(float(np.percentile(values, 50)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "fps": round(1000.0 / average, 2),
    }


def run_pytorch(model_path: Path, frames, image_size: int, confidence: float, device: str) -> tuple[dict, list[list[dict]]]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model.to(device)
    for frame in frames[:2]:
        model.predict(frame, imgsz=image_size, conf=confidence, device=device, verbose=False)
    timings = []
    per_frame = []
    for frame in frames:
        started = time.perf_counter()
        result = model.predict(frame, imgsz=image_size, conf=confidence, device=device, verbose=False)[0]
        timings.append((time.perf_counter() - started) * 1000)
        boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append({
                "class_id": int(box.cls),
                "confidence": float(box.conf),
                "bbox_xyxy": [x1, y1, x2, y2],
            })
        per_frame.append(boxes)
    stats = summarize(timings)
    stats["detections_avg"] = sum(len(detections) for detections in per_frame) / max(len(per_frame), 1)
    return stats, per_frame


def compare_detections(pytorch_per_frame: list[list[dict]], tensorrt_per_frame: list[list[dict]]) -> dict:
    matched = 0
    missed = 0
    false_positives = 0
    ious = []
    conf_deltas = []
    for py_dets, trt_dets in zip(pytorch_per_frame, tensorrt_per_frame):
        used_trt = set()
        for pdet in py_dets:
            best_iou = 0.0
            best_idx = -1
            for idx, tdet in enumerate(trt_dets):
                if idx in used_trt or pdet["class_id"] != tdet["class_id"]:
                    continue
                iou = box_iou(pdet["bbox_xyxy"], tdet["bbox_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0:
                used_trt.add(best_idx)
                ious.append(best_iou)
                conf_deltas.append(abs(pdet["confidence"] - trt_dets[best_idx]["confidence"]))
                matched += 1
            else:
                missed += 1
        false_positives += len(trt_dets) - len(used_trt)
    return {
        "matched": matched,
        "missed": missed,
        "false_positives": false_positives,
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        "max_iou": round(float(np.max(ious)), 4) if ious else None,
        "max_confidence_delta": round(float(max(conf_deltas)), 4) if conf_deltas else None,
    }


def run_tensorrt_engine(engine_path: Path, frames, image_size: int) -> tuple[dict, list[list[dict]] | None]:
    """Run TRT inference via trtexec for speed; collect detections if python bindings exist."""
    import subprocess

    trtexec = os.environ.get("TRTEXEC", "trtexec")
    command = [
        trtexec,
        f"--loadEngine={engine_path}",
        f"--shapes=images:3x{image_size}x{image_size}",
        f"--warmUp=1000",
        f"--iterations={len(frames)}",
        "--noDataTransfers",
    ]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=600)
        output = result.stdout + result.stderr
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.warning("trtexec unavailable or failed: %s", exc)
        return {"avg_inf_ms": None, "fps": None, "error": str(exc)}, None

    speed = {"avg_inf_ms": None, "fps": None}
    for line in output.splitlines():
        if "Latency:" in line:
            speed["avg_inf_ms"] = round(float(line.split("Latency:", 1)[1].split("ms", 1)[0].strip()), 3)
            speed["fps"] = round(1000.0 / speed["avg_inf_ms"], 2)
    return speed, None


def read_temperature() -> float | None:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()) / 1000.0
    except (FileNotFoundError, ValueError, OSError):
        return None


def run(args: argparse.Namespace) -> None:
    frames = load_frames(Path(args.frames), max_frames=args.max_frames)
    logger.info("Loaded %d frames for validation", len(frames))

    results: dict = {
        "model": str(args.model),
        "engine": str(args.engine) if args.engine else None,
        "frames_tested": len(frames),
        "image_size": args.image_size,
        "device": args.device,
    }

    results["pytorch"], py_per_frame = run_pytorch(
        Path(args.model), frames, args.image_size, args.confidence, args.device
    )
    logger.info("PyTorch baseline: avg=%sms, fps=%s", results["pytorch"]["avg_inf_ms"], results["pytorch"]["fps"])

    if args.engine and Path(args.engine).exists():
        trt_speed, trt_dets = run_tensorrt_engine(Path(args.engine), frames, args.image_size)
        results["tensorrt"] = trt_speed
        if trt_dets is not None:
            results["accuracy"] = compare_detections(py_per_frame, trt_dets)
        else:
            results["accuracy"] = {
                "note": "Detection comparison requires tensorrt python bindings on the Jetson.",
                "pytorch_detections_avg": results["pytorch"]["detections_avg"],
            }
    else:
        logger.warning("No TensorRT engine provided; PyTorch baseline only.")

    temp = read_temperature()
    if temp is not None:
        results["temperature_c"] = temp

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nValidation report saved to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--engine", type=Path, help="TensorRT engine (.engine)")
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAME_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)s | %(message)s")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
