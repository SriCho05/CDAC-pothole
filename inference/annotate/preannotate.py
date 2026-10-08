"""Batch pre-annotation: run the local YOLO model over images or video frames."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from PIL import Image

try:
    from .classes import CLASS_IDS, CLASS_NAMES
    from .schema import Detection, FrameRecord, write_json_atomic, write_yolo_label
except ImportError:  # Running directly as: python preannotate.py
    from classes import CLASS_IDS, CLASS_NAMES
    from schema import Detection, FrameRecord, write_json_atomic, write_yolo_label

logger = logging.getLogger("annotate_preannotate")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def unique_frame_id(stem: str, used: set[str]) -> str:
    frame_id = stem
    counter = 2
    while frame_id in used:
        frame_id = f"{stem}_{counter}"
        counter += 1
    used.add(frame_id)
    return frame_id


def collect_images(source: Path, patterns: str | None) -> list[Path]:
    candidates = []
    if patterns:
        for pattern in patterns.split(","):
            candidates.extend(source.glob(pattern.strip()))
    else:
        candidates.extend(source.iterdir())
    return sorted({path for path in candidates if path.is_file() and path.suffix.lower() in IMAGE_EXTS})


def build_manifest_images(source: Path, patterns: str | None) -> dict:
    manifest = {}
    used: set[str] = set()
    for path in collect_images(source, patterns):
        frame_id = unique_frame_id(path.stem, used)
        with Image.open(path) as image:
            width, height = image.size
        manifest[frame_id] = {
            "path": str(path.resolve()),
            "width": width,
            "height": height,
            "timestamp_unix_us": int(path.stat().st_mtime * 1_000_000),
        }
    return manifest


def build_manifest_video(source: Path, out_dir: Path, every: int) -> dict:
    import cv2

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    base_timestamp = int(source.stat().st_mtime * 1_000_000)
    manifest = {}
    index = 0
    saved = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % max(1, every) == 0:
            frame_id = f"{source.stem}_{saved:06d}"
            path = frames_dir / f"{frame_id}.jpg"
            cv2.imwrite(str(path), frame)
            manifest[frame_id] = {
                "path": str(path.resolve()),
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "timestamp_unix_us": base_timestamp + int(index / fps * 1_000_000),
            }
            saved += 1
        index += 1
    capture.release()
    return manifest


def detect_frame(model, path: str, image_size: int, confidence: float, device: str) -> tuple[list[Detection], int]:
    image = Image.open(path).convert("RGB")
    results = model.predict(image, imgsz=image_size, conf=confidence, device=device, verbose=False)
    detections = []
    unmapped = 0
    for result in results:
        for box in result.boxes:
            name = str(result.names[int(box.cls)])
            class_id = CLASS_IDS.get(name)
            if class_id is None:
                unmapped += 1
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            detections.append(Detection.from_xyxy(class_id, [x1, y1, x2, y2], confidence=float(box.conf)))
    return detections, unmapped


def run(args: argparse.Namespace) -> None:
    source = Path(args.source)
    out_dir = Path(args.out)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        manifest = build_manifest_images(source, args.pattern)
    elif source.is_file() and source.suffix.lower() in VIDEO_EXTS:
        manifest = build_manifest_video(source, out_dir, args.every)
    else:
        raise SystemExit(f"Source must be an image directory or video file: {source}")

    if not manifest:
        raise SystemExit(f"No frames found in {source}")
    write_json_atomic(out_dir / "manifest.json", manifest)

    import torch
    from ultralytics import YOLO

    model_path = os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent.parent / "best.pt"))
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)
    if Path(model_path).suffix.lower() == ".pt":
        model.to(device)
    logger.info("Model %s on %s", model_path, device)

    records = {}
    total_detections = 0
    unmapped = 0
    started = time.perf_counter()
    for position, (frame_id, entry) in enumerate(manifest.items(), start=1):
        detections, unmapped_here = detect_frame(
            model, entry["path"], args.image_size, args.confidence, device
        )
        unmapped += unmapped_here
        record = FrameRecord(
            frame_id=frame_id,
            timestamp_unix_us=entry["timestamp_unix_us"],
            image_width=entry["width"],
            image_height=entry["height"],
            source_path=entry["path"],
            detections=detections,
        )
        write_yolo_label(labels_dir / f"{frame_id}.txt", record)
        records[frame_id] = record.to_dict()
        total_detections += len(detections)
        if position % 100 == 0 or position == len(manifest):
            elapsed = max(time.perf_counter() - started, 1e-9)
            logger.info("Pre-annotated %d/%d frames (%.1f fps)", position, len(manifest), position / elapsed)

    write_json_atomic(out_dir / "predictions.json", records)
    histogram = {}
    for record in records.values():
        for detection in record["detections"]:
            name = CLASS_NAMES.get(detection["class_id"], detection["class_id"])
            histogram[name] = histogram.get(name, 0) + 1
    print(f"Workspace:      {out_dir.resolve()}")
    print(f"Frames:         {len(records)}")
    print(f"Detections:     {total_detections}")
    print(f"Class histogram: {histogram}")
    if unmapped:
        print(f"Skipped {unmapped} box(es) whose class is not in the annotation registry.")
    print("Next: python -m annotate.annotation_app  (open http://127.0.0.1:8765)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-annotate frames with the local YOLO model")
    parser.add_argument("--source", required=True, help="Image directory or video file")
    parser.add_argument("--out", default="annotations", help="Annotation workspace directory")
    parser.add_argument("--conf", type=float, default=float(os.getenv("CONFIDENCE", "0.25")))
    parser.add_argument("--image-size", type=int, default=int(os.getenv("IMAGE_SIZE", "640")))
    parser.add_argument("--device", default=os.getenv("YOLO_DEVICE"), help="cuda:0 or cpu")
    parser.add_argument("--pattern", default=None, help="Comma-separated glob patterns, e.g. '*.jpg,*.png'")
    parser.add_argument("--every", type=int, default=10, help="Video: keep every Nth frame")
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
