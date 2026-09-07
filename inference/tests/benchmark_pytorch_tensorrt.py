#!/usr/bin/env python3
"""Compare PyTorch and TensorRT using a camera-free synthetic input."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


def create_test_image(image_size: int) -> np.ndarray:
    import cv2

    image = np.random.default_rng(7).integers(
        0, 255, (image_size, image_size, 3), dtype=np.uint8
    )
    cv2.rectangle(image, (50, 50), (200, 200), (100, 100, 100), -1)
    cv2.circle(image, (image_size * 2 // 3, image_size // 2), 50, (50, 50, 50), -1)
    cv2.line(image, (0, image_size // 2), (image_size, image_size // 2), (100, 100, 100), 2)
    return image


def benchmark_pytorch(model_path: Path, image: np.ndarray, device: str, warmup: int, iterations: int) -> dict:
    import torch
    from ultralytics import YOLO

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = YOLO(str(model_path))
    for _ in range(warmup):
        model.predict(image, device=device, verbose=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        model.predict(image, device=device, verbose=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000)
    return summarize(timings)


def benchmark_tensorrt(engine_path: Path, image_size: int, warmup: int, iterations: int, trtexec: str) -> dict:
    command = [
        trtexec,
        f"--loadEngine={engine_path}",
        f"--shapes=images:1x3x{image_size}x{image_size}",
        f"--warmUp={warmup * 100}",
        f"--iterations={iterations}",
        "--noDataTransfers",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    output = completed.stdout + completed.stderr
    for line in output.splitlines():
        if "Latency:" in line:
            latency = float(line.split("Latency:", 1)[1].split("ms", 1)[0].strip())
            return {"avg_time_ms": latency, "fps": 1000.0 / latency}
    raise RuntimeError("trtexec completed without reporting latency")


def summarize(timings: list[float]) -> dict:
    values = np.asarray(timings)
    average = float(values.mean())
    return {
        "avg_time_ms": average,
        "std_time_ms": float(values.std()),
        "min_time_ms": float(values.min()),
        "max_time_ms": float(values.max()),
        "fps": 1000.0 / average,
    }


def export_engine(model_path: Path, output_dir: Path, image_size: int) -> Path:
    from ultralytics import YOLO

    test_model = output_dir / model_path.name
    shutil.copy2(model_path, test_model)
    exported = Path(YOLO(str(test_model)).export(format="engine", imgsz=image_size, half=True, device=0))
    engine_path = output_dir / "pothole_fp16.engine"
    if exported != engine_path:
        shutil.move(str(exported), engine_path)
    return engine_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("inference/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inference/tests/artifacts"))
    parser.add_argument("--mode", choices=("pytorch", "tensorrt", "both"), default="both")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trtexec", default="/usr/src/tensorrt/bin/trtexec")
    parser.add_argument("--export", action="store_true", help="Export a test-only TensorRT engine")
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = create_test_image(args.image_size)
    results = {"model": str(args.model), "image_size": args.image_size, "device": args.device}

    if args.mode in ("pytorch", "both"):
        results["pytorch"] = benchmark_pytorch(args.model, image, args.device, args.warmup, args.iterations)
    if args.mode in ("tensorrt", "both"):
        engine = args.engine
        if args.export:
            engine = export_engine(args.model, args.output_dir, args.image_size)
        if engine is None or not engine.is_file():
            raise FileNotFoundError("Provide --engine or use --export for the TensorRT run")
        results["tensorrt"] = benchmark_tensorrt(engine, args.image_size, args.warmup, args.iterations, args.trtexec)

    result_path = args.output_dir / "comparison.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Results saved to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())