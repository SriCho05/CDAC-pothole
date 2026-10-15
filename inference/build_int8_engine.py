"""Build an INT8 TensorRT engine with proper image-based calibration.

Uses the Kaggle pothole dataset images in inference/calib_images/ for
calibration. Produces best_int8.engine and calibration.cache.

Usage:
    python -m build_int8_engine \\
        --onnx inference/best.onnx \\
        --calib-dir inference/calib_images \\
        --output inference/best_int8.engine \\
        --batch-size 1 \\
        --image-size 640
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
import tensorrt as trt
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("build_int8")

INFERENCE_DIR = Path(__file__).resolve().parent
DEFAULT_ONNX = INFERENCE_DIR / "best.onnx"
DEFAULT_CALIB_DIR = INFERENCE_DIR / "calib_images"
DEFAULT_OUTPUT = INFERENCE_DIR / "best_int8.engine"
CACHE_FILE = INFERENCE_DIR / "calibration.cache"

INPUT_NAME = "images"
IMAGE_SIZE = 640


class ImageBatchStream(trt.IInt8EntropyCalibrator2):
    """Entropy calibrator that reads batches of images for INT8 calibration."""

    def __init__(self, calibration_files, batch_size, cache_file):
        super().__init__()
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.max_batches = len(calibration_files) // batch_size
        self.files = calibration_files[: self.max_batches * batch_size]
        self.current_index = 0

        self.device_input = cuda.mem_alloc(
            trt.volume((batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)) * np.float32().itemsize
        )
        logger.info("Calibrator: %d images, %d batches", len(self.files), self.max_batches)

    def get_batch_size(self) -> int:
        return self.batch_size

    def _preprocess(self, img_path: str) -> np.ndarray:
        """Preprocess image the same way YOLO does: resize to 640x640,
        BGR->RGB, normalize to [0,1], CHW."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return img

    def get_batch(self, names):
        batch = np.empty((self.batch_size, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        for i in range(self.batch_size):
            idx = self.current_index + i
            if idx >= len(self.files):
                return None
            batch[i] = self._preprocess(self.files[idx])
        self.current_index += self.batch_size
        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        logger.info("Wrote calibration cache: %s", self.cache_file)


def build_int8_engine(
    onnx_path: str,
    calib_files: list,
    output_path: str,
    batch_size: int,
    workspace_size: int = 2048,
) -> str:
    """Build an INT8 TensorRT engine from ONNX with image-based calibration."""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(parser.get_error(i))
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size << 20)
    config.set_flag(trt.BuilderFlag.INT8)

    # Create optimization profile for dynamic input
    profile = builder.create_optimization_profile()
    profile.set_shape(INPUT_NAME, (1, 3, IMAGE_SIZE, IMAGE_SIZE),
                      (1, 3, IMAGE_SIZE, IMAGE_SIZE),
                      (batch_size, 3, IMAGE_SIZE, IMAGE_SIZE))
    config.add_optimization_profile(profile)
    config.set_calibration_profile(profile)

    # Create calibrator
    calibrator = ImageBatchStream(calib_files, batch_size, CACHE_FILE)
    config.int8_calibrator = calibrator

    logger.info("Building INT8 engine (this may take a while)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        logger.error("Failed to build INT8 engine")
        sys.exit(1)

    with open(output_path, "wb") as f:
        f.write(serialized)
    logger.info("Engine saved: %s (%.1f MB)", output_path, os.path.getsize(output_path) / 1e6)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build INT8 TensorRT engine with image calibration")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX), help="Path to ONNX model")
    parser.add_argument("--calib-dir", default=str(DEFAULT_CALIB_DIR), help="Calibration images directory")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output engine path")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Input image size")
    parser.add_argument("--workspace", type=int, default=2048, help="Workspace size in MB")
    parser.add_argument("--num-calib", type=int, default=500, help="Number of calibration images to use")
    parser.add_argument("--calib-batch-size", type=int, default=16, help="Batch size for calibration")
    args = parser.parse_args()

    calib_files = sorted(glob.glob(os.path.join(args.calib_dir, "*.jpg")))
    if not calib_files:
        logger.error("No calibration images found in %s", args.calib_dir)
        sys.exit(1)
    calib_files = calib_files[: args.num_calib]
    logger.info("Using %d calibration images from %s", len(calib_files), args.calib_dir)

    if CACHE_FILE.exists():
        logger.info("Calibration cache exists, re-using: %s", CACHE_FILE)

    build_int8_engine(
        onnx_path=args.onnx,
        calib_files=calib_files,
        output_path=args.output,
        batch_size=args.calib_batch_size,
        workspace_size=args.workspace,
    )


if __name__ == "__main__":
    main()
