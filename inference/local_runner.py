"""Run pothole detection entirely on a Jetson-connected camera."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

try:
    from .camera import open_camera, open_video
    from .gnss import GnssReader
except ImportError:  # Running directly as: python local_runner.py
    from camera import open_camera, open_video
    from gnss import GnssReader

logger = logging.getLogger("jetson_local_runner")


def env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "on")


def save_event(db_path: str, uploads_dir: Path, frame, cv2, label: str, confidence: float, position) -> str:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    filename = f"jetson_{int(time.time() * 1000)}.jpg"
    output = uploads_dir / filename
    cv2.imwrite(str(output), frame)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS potholes (id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT, confidence REAL, latitude REAL, longitude REAL, timestamp TEXT, image_path TEXT)"
        )
        conn.execute(
            "INSERT INTO potholes (label, confidence, latitude, longitude, timestamp, image_path) VALUES (?, ?, ?, ?, ?, ?)",
            (label, confidence, position.latitude if position else None, position.longitude if position else None,
             datetime.now(timezone.utc).isoformat(), f"/uploads/{filename}"),
        )
        conn.commit()
    return str(output)


def write_json(path: Path, payload: dict) -> None:
    """Write status atomically so the API never reads half-written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    requested_device = os.getenv("YOLO_DEVICE")
    if requested_device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        warnings.filterwarnings(
            "ignore",
            message=r"CUDA initialization: The NVIDIA driver on your system is too old.*",
            category=UserWarning,
        )

    import cv2
    import torch
    from ultralytics import YOLO

    model_path = os.getenv("MODEL_PATH", str(Path(__file__).with_name("best.pt")))
    device = requested_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)
    model.to(device)
    capture = (open_video(cv2, args.video) if args.video else
               open_camera(cv2, args.camera_type, args.camera_device, args.width, args.height, args.fps))
    gnss = GnssReader(args.gnss_port, args.gnss_baud) if args.gnss_port else None
    if gnss:
        gnss.start()

    db_path = os.getenv("DB_PATH", str(Path(__file__).with_name("potholes.db")))
    uploads_dir = Path(os.getenv("UPLOADS_DIR", str(Path(__file__).with_name("uploads"))))
    uploads_dir.mkdir(parents=True, exist_ok=True)
    status_path = Path(os.getenv("STATUS_PATH", str(Path(__file__).with_name("local_status.json"))))
    latest_frame_path = Path(os.getenv("LATEST_FRAME_PATH", str(uploads_dir / "latest.jpg")))
    last_saved = 0.0
    frames = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if args.video:
                    logger.info("Video finished")
                    break
                logger.warning("Camera frame unavailable")
                continue
            frame_started = time.perf_counter()
            result = model.predict(frame, imgsz=args.image_size, conf=args.confidence, verbose=False)[0]
            potholes = []
            for box in result.boxes:
                class_id = int(box.cls)
                name = str(result.names[class_id])
                label = "pothole" if name == "0" or class_id == 0 else name
                detection = {"label": label, "confidence": float(box.conf), "box": box.xyxy[0].tolist()}
                if label == "pothole":
                    potholes.append(detection)
                    x1, y1, x2, y2 = map(int, detection["box"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"pothole {detection['confidence']:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            now = time.time()
            position = gnss.snapshot() if gnss else None
            if potholes and now - last_saved >= args.save_cooldown:
                save_event(db_path, uploads_dir, frame, cv2, "pothole", max(d["confidence"] for d in potholes), position)
                last_saved = now
            frames += 1
            elapsed = time.perf_counter() - started
            if args.latest_frame_interval > 0 and frames % args.latest_frame_interval == 0:
                cv2.imwrite(str(latest_frame_path), frame)
            write_json(status_path, {
                "status": "running", "camera_type": args.camera_type,
                "input": args.video or args.camera_type, "device": device,
                "frames": frames, "fps": round(frames / elapsed, 2),
                "last_frame_ms": round((time.perf_counter() - frame_started) * 1000, 2),
                "detections": len(potholes), "gnss_fix": position is not None,
                "latest_frame": "/uploads/latest.jpg" if args.latest_frame_interval > 0 else None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            if args.display:
                cv2.imshow("Pothole Detection - Jetson", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if gnss:
            gnss.stop()
        cv2.destroyAllWindows()
        write_json(status_path, {"status": "stopped", "updated_at": datetime.now(timezone.utc).isoformat()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Local pothole detector for a camera or video")
    parser.add_argument("--video", default=os.getenv("VIDEO_PATH"), help="Video file to process instead of a camera")
    parser.add_argument("--camera-type", default=os.getenv("CAMERA_TYPE", "csi"), choices=("usb", "csi"))
    parser.add_argument("--camera-device", default=os.getenv("CAMERA_DEVICE", "0"))
    parser.add_argument("--width", type=int, default=int(os.getenv("CAMERA_WIDTH", "1280")))
    parser.add_argument("--height", type=int, default=int(os.getenv("CAMERA_HEIGHT", "720")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("CAMERA_FPS", "30")))
    parser.add_argument("--image-size", type=int, default=int(os.getenv("IMAGE_SIZE", "640")))
    parser.add_argument("--confidence", type=float, default=float(os.getenv("CONFIDENCE", "0.25")))
    parser.add_argument("--save-cooldown", type=float, default=float(os.getenv("SAVE_COOLDOWN", "5")))
    parser.add_argument("--latest-frame-interval", type=int, default=int(os.getenv("LATEST_FRAME_INTERVAL", "5")))
    parser.add_argument("--gnss-port", default=os.getenv("GNSS_PORT", "/dev/ttyUSB0"))
    parser.add_argument("--gnss-baud", type=int, default=int(os.getenv("GNSS_BAUD", "9600")))
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=env_bool("DISPLAY", True))
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()