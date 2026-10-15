"""Run INT8 TensorRT engine directly on a 60 FPS video for maximum performance.

Bypasses the Ultralytics wrapper to reduce preprocessing/post-processing overhead.

Usage:
    python inference/run_int8_video_trt.py \\
        --engine inference/best_int8.engine \\
        --video "/path/to/demo 60fps.mp4" \\
        --confidence 0.25
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
import tensorrt as trt

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("int8_trt")

IMAGE_SIZE = 640


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Resize to 640x640, BGR->RGB, normalize to [0,1], CHW, add batch dim."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return np.ascontiguousarray(np.expand_dims(img, axis=0))


class FrameReader:
    """Background thread that reads and preprocesses frames to overlap with GPU inference."""

    def __init__(self, video_path: str, max_queue: int = 4):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self.queue = queue.Queue(maxsize=max_queue)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                self.queue.put(None)
                break
            input_data = preprocess(frame)
            self.queue.put((frame, input_data))

    def start(self):
        self._thread.start()

    def read(self):
        return self.queue.get()

    def stop(self):
        self._stop = True
        self._thread.join(timeout=2)
        self.cap.release()


def postprocess(pred: np.ndarray, conf_thres: float, scale_x: float, scale_y: float) -> list:
    """Decode YOLO output and apply NMS.

    pred shape: (1, 5, 8400) -> per anchor: [x_center, y_center, width, height, conf]
    Bbox values are in 640x640 pixel coordinates.
    """
    pred = pred[0]  # (5, 8400)
    pred = np.transpose(pred, (1, 0))  # (8400, 5)

    # Filter by confidence (vectorized)
    confs = pred[:, 4]
    mask = confs >= conf_thres
    if not np.any(mask):
        return []

    filtered = pred[mask]
    confs_filtered = filtered[:, 4]

    cxs = filtered[:, 0]
    cys = filtered[:, 1]
    ws = filtered[:, 2]
    hs = filtered[:, 3]

    x1s = (cxs - ws / 2) * scale_x
    y1s = (cys - hs / 2) * scale_y
    x2s = (cxs + ws / 2) * scale_x
    y2s = (cys + hs / 2) * scale_y

    boxes = np.column_stack([x1s, y1s, x2s, y2s]).astype(np.float32)
    scores = confs_filtered.astype(np.float32)

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, 0.45)
    if len(indices) == 0:
        return []

    result = []
    for idx in indices.flatten():
        x1, y1, x2, y2 = boxes[idx]
        result.append({
            "bbox": [int(max(0, x1)), int(max(0, y1)), int(min(65535, x2)), int(min(65535, y2))],
            "confidence": float(scores[idx]),
            "class_id": 0,
        })
    return result


class TRTInference:
    """Direct TensorRT engine inference (TRT 10 API)."""

    def __init__(self, engine_path: str, image_size: int = IMAGE_SIZE):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Get tensor names
        self.input_name = "images"
        self.output_name = "output0"

        # Allocate buffers once
        input_shape = (1, 3, image_size, image_size)
        output_shape = self.context.get_tensor_shape(self.output_name)

        self.input_size = trt.volume(input_shape) * np.float32().itemsize
        self.output_size = trt.volume(output_shape) * np.float32().itemsize
        self.output_shape = output_shape

        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        self.context.set_tensor_address(self.input_name, self.d_input)
        self.context.set_tensor_address(self.output_name, self.d_output)
        self.stream = cuda.Stream()

        self.host_output = np.empty(output_shape, dtype=np.float32)

        logger.info("Engine loaded: %s", engine_path)
        logger.info("Input: %s, Output: %s", input_shape, output_shape)

    def infer(self, input_data: np.ndarray):
        """Run inference with pre-processed input, return raw output."""
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        self.context.execute_async_v3(stream_handle=int(self.stream.handle))
        cuda.memcpy_dtoh_async(self.host_output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.host_output


def run(engine_path: str, video_path: str, confidence: float, output_path: str,
        save_video: str = None, display: bool = False) -> None:
    trt_infer = TRTInference(engine_path)
    logger.info("Direct TensorRT engine ready")

    reader = FrameReader(video_path, max_queue=8)
    total_frames = reader.total_frames
    video_fps = reader.video_fps
    width = reader.width
    height = reader.height
    scale_x = width / IMAGE_SIZE
    scale_y = height / IMAGE_SIZE
    logger.info("Video: %d frames, %.1f fps, %dx%d", total_frames, video_fps, width, height)

    reader.start()

    out_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(save_video, fourcc, video_fps, (width, height))
        logger.info("Saving output video: %s", save_video)

    frame_count = 0
    detection_total = 0
    timings = []
    inference_only = []
    start = time.perf_counter()

    while True:
        item = reader.read()
        if item is None:
            logger.info("Video ended at frame %d/%d", frame_count, total_frames)
            break

        frame, input_data = item

        t0 = time.perf_counter()
        output = trt_infer.infer(input_data)
        infer_ms = (time.perf_counter() - t0) * 1000
        inference_only.append(infer_ms)

        detections = postprocess(output, confidence, scale_x, scale_y)
        detection_total += len(detections)
        frame_ms = (time.perf_counter() - t0) * 1000
        timings.append(frame_ms)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"pothole {det['confidence']:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if save_video:
            out_writer.write(frame)
        if display:
            cv2.imshow("INT8 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.perf_counter() - start
            wall_fps = frame_count / elapsed
            avg_frame_ms = float(np.mean(timings))
            avg_infer_ms = float(np.mean(inference_only))
            infer_fps = 1000.0 / avg_infer_ms
            logger.info(
                "Frame %d/%d | %.1f%% | wall %.1f FPS | infer %.1fms = %.1f FPS | det=%d",
                frame_count, total_frames, 100 * frame_count / total_frames,
                wall_fps, avg_infer_ms, infer_fps, len(detections),
            )

    reader.stop()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    wall_fps = frame_count / elapsed
    frame_np = np.asarray(timings)
    infer_np = np.asarray(inference_only)

    report = {
        "engine": engine_path,
        "video": video_path,
        "video_fps": round(video_fps, 2),
        "video_resolution": f"{width}x{height}",
        "total_frames_in_video": total_frames,
        "frames_processed": frame_count,
        "wall_time_sec": round(elapsed, 2),
        "wall_fps": round(wall_fps, 2),
        "inference_only_avg_ms": round(float(infer_np.mean()), 3),
        "inference_only_fps": round(1000.0 / float(infer_np.mean()), 2),
        "inference_only_p95_ms": round(float(np.percentile(infer_np, 95)), 3),
        "total_processing_avg_ms": round(float(frame_np.mean()), 3),
        "total_processing_fps": round(1000.0 / float(frame_np.mean()), 2),
        "total_detections": detection_total,
        "detections_per_frame": round(detection_total / max(frame_count, 1), 2),
    }

    logger.info("=== RESULTS ===")
    logger.info("Frames processed: %d/%d", frame_count, total_frames)
    logger.info("Wall-clock FPS: %.2f", wall_fps)
    logger.info("Inference-only: avg=%.1f ms = %.1f FPS, p95=%.1f ms",
                report["inference_only_avg_ms"], report["inference_only_fps"],
                report["inference_only_p95_ms"])
    logger.info("Total processing: avg=%.1f ms = %.1f FPS",
                report["total_processing_avg_ms"],
                report["total_processing_fps"])
    logger.info("Total detections: %d (%.2f per frame)", detection_total, report["detections_per_frame"])

    out = Path(output_path)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nResults saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct TensorRT INT8 inference on video")
    parser.add_argument("--engine", required=True, help="TensorRT INT8 engine path")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output", default="int8_video_results_trt.json")
    parser.add_argument("--save-video", default="int8_output.mp4", help="Save output video with detections")
    parser.add_argument("--display", action="store_true", help="Show window with detections")
    args = parser.parse_args()
    run(args.engine, args.video, args.confidence, args.output, args.save_video, args.display)


if __name__ == "__main__":
    main()
