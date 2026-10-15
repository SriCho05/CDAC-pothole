"""Run INT8 TensorRT engine on a 60 FPS video for real-world performance testing.

Usage:
    python inference/run_int8_video.py \
        --engine inference/best_int8.engine \
        --video "/path/to/demo 60fps.mp4" \
        --confidence 0.25 \
        --output inference/int8_video_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("int8_video")


def run(engine_path: str, video_path: str, confidence: float, output_path: str) -> None:
    model = YOLO(engine_path, verbose=True)
    logger.info("Loaded INT8 engine: %s", engine_path)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Video: %d frames, %.1f fps, %dx%d", total_frames, video_fps, width, height)

    frame_count = 0
    detection_total = 0
    timings = []
    start = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            logger.info("Video ended at frame %d/%d", frame_count, total_frames)
            break

        t0 = time.perf_counter()
        results = model.predict(frame, imgsz=640, conf=confidence, verbose=False)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        timings.append(elapsed_ms)

        boxes = results.boxes
        detection_total += len(boxes)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"pothole {conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.perf_counter() - start
            current_fps = frame_count / elapsed
            avg_ms = float(np.mean(timings))
            infer_fps = 1000.0 / avg_ms
            logger.info(
                "Frame %d/%d | %.1f%% | %.1f fps (infer %.1fms = %.1f FPS) | detections=%d",
                frame_count, total_frames, 100 * frame_count / total_frames,
                current_fps, avg_ms, infer_fps, len(boxes),
            )

    capture.release()
    cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    wall_fps = frame_count / elapsed
    timings_np = np.asarray(timings)
    report = {
        "engine": engine_path,
        "video": video_path,
        "video_fps": round(video_fps, 2),
        "video_resolution": f"{width}x{height}",
        "total_frames_in_video": total_frames,
        "frames_processed": frame_count,
        "wall_time_sec": round(elapsed, 2),
        "wall_fps": round(wall_fps, 2),
        "inference_avg_ms": round(float(timings_np.mean()), 3),
        "inference_median_ms": round(float(np.median(timings_np)), 3),
        "inference_p95_ms": round(float(np.percentile(timings_np, 95)), 3),
        "inference_fps": round(1000.0 / float(timings_np.mean()), 2),
        "total_detections": detection_total,
        "detections_per_frame": round(detection_total / max(frame_count, 1), 2),
    }

    logger.info("=== RESULTS ===")
    logger.info("Frames processed: %d/%d", frame_count, total_frames)
    logger.info("Wall-clock FPS: %.2f", wall_fps)
    logger.info("Inference avg: %.1f ms (%.1f FPS)", report["inference_avg_ms"], report["inference_fps"])
    logger.info("Inference p95: %.1f ms", report["inference_p95_ms"])
    logger.info("Total detections: %d (%.2f per frame)", detection_total, report["detections_per_frame"])

    out = Path(output_path)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nResults saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run INT8 TensorRT engine on video")
    parser.add_argument("--engine", required=True, help="TensorRT INT8 engine path")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output", default="int8_video_results.json")
    args = parser.parse_args()
    run(args.engine, args.video, args.confidence, args.output)


if __name__ == "__main__":
    main()
