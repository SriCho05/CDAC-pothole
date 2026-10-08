"""Local annotation and reannotation server for road-anomaly frames."""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

try:
    import cv2
except ImportError:  # annotated rendering is optional
    cv2 = None

try:
    from .classes import CLASS_NAMES, SEVERITY_LEVELS
    from .schema import Detection, FrameRecord, build_coco, read_yolo_label, write_json_atomic, write_yolo_label, xywh_to_xyxy
except ImportError:  # Running directly as: python annotation_app.py
    from classes import CLASS_NAMES, SEVERITY_LEVELS
    from schema import Detection, FrameRecord, build_coco, read_yolo_label, write_json_atomic, write_yolo_label, xywh_to_xyxy

logger = logging.getLogger("annotate_app")

INFERENCE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE = INFERENCE_DIR / "annotations"
UI_PATH = Path(__file__).resolve().with_name("annotate.html")

CLASS_PALETTE_BGR = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
    (255, 0, 0), (255, 0, 255), (0, 128, 255), (0, 200, 200), (128, 128, 128),
]


def render_annotated(record: FrameRecord) -> bytes | None:
    """Draw detections on the source frame and return JPEG bytes."""
    if cv2 is None or not record.source_path:
        return None
    image = cv2.imread(record.source_path)
    if image is None:
        return None
    for detection in record.detections:
        x1, y1, x2, y2 = map(int, xywh_to_xyxy(detection.bbox_xywh))
        color = CLASS_PALETTE_BGR[detection.class_id % len(CLASS_PALETTE_BGR)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f} {detection.severity}"
        cv2.putText(image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buffer.tobytes() if ok else None


class Workspace:
    def __init__(self, root: Path):
        self.root = root
        self.labels_dir = root / "labels"
        self.manifest_path = root / "manifest.json"
        self.predictions_path = root / "predictions.json"
        self.lock = threading.Lock()
        self._label_cache: dict[str, tuple[int, list[Detection]]] = {}
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._read_json(self.manifest_path, {})
        self.records = self._read_json(self.predictions_path, {})

    @staticmethod
    def _read_json(path: Path, default):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return default

    def _persist(self) -> None:
        write_json_atomic(self.predictions_path, self.records)

    def _read_label(self, frame_id: str, width: int, height: int) -> list[Detection]:
        path = self.labels_dir / f"{frame_id}.txt"
        mtime = path.stat().st_mtime_ns if path.exists() else None
        cached = self._label_cache.get(frame_id)
        if cached and cached[0] == mtime:
            return cached[1]
        detections = read_yolo_label(path, width, height) if path.exists() else []
        self._label_cache[frame_id] = (mtime, detections)
        return detections

    def summaries(self, filter_value: str = "all") -> dict:
        frames = []
        labeled_total = 0
        detection_total = 0
        for frame_id, entry in self.manifest.items():
            record = self.get_record(frame_id)
            labeled = (self.labels_dir / f"{frame_id}.txt").exists()
            count = len(record.detections)
            labeled_total += int(labeled)
            detection_total += count
            if filter_value == "labeled" and not labeled:
                continue
            if filter_value == "unlabeled" and labeled:
                continue
            frames.append({
                "frame_id": frame_id,
                "labeled": labeled,
                "detections": count,
                "width": entry.get("width", 0),
                "height": entry.get("height", 0),
            })
        return {
            "root": str(self.root),
            "total": len(self.manifest),
            "labeled": labeled_total,
            "detections": detection_total,
            "frames": frames,
        }

    def get_record(self, frame_id: str) -> FrameRecord | None:
        entry = self.manifest.get(frame_id)
        if entry is None:
            return None
        width = int(entry.get("width", 0))
        height = int(entry.get("height", 0))
        stored = self.records.get(frame_id)
        if stored is not None:
            # predictions.json holds the full detection attributes (severity,
            # confidence, depth, area) that the 5-column YOLO label cannot store.
            record = FrameRecord.from_dict(stored)
        else:
            record = FrameRecord.from_dict({
                "frame_id": frame_id,
                "timestamp_unix_us": entry.get("timestamp_unix_us", 0),
                "image_width": width,
                "image_height": height,
                "source_path": entry.get("path"),
            })
            record.detections = self._read_label(frame_id, width, height)
        if not record.image_width:
            record.image_width = width
        if not record.image_height:
            record.image_height = height
        return record

    def save_detections(self, frame_id: str, detections: list[Detection]) -> FrameRecord:
        record = self.get_record(frame_id)
        if record is None:
            raise KeyError(frame_id)
        record.detections = [detection for detection in detections if detection.bbox_xywh[2] > 0 and detection.bbox_xywh[3] > 0]
        self.records[frame_id] = record.to_dict()
        write_yolo_label(self.labels_dir / f"{frame_id}.txt", record)
        with self.lock:
            self._persist()
        return record

    def save_meta(self, frame_id: str, telemetry: dict, environment: dict) -> FrameRecord:
        record = self.get_record(frame_id)
        if record is None:
            raise KeyError(frame_id)
        record.telemetry.update(telemetry)
        record.environment.update(environment)
        self.records[frame_id] = record.to_dict()
        with self.lock:
            self._persist()
        return record

    def clear_labels(self, frame_id: str) -> None:
        record = self.get_record(frame_id)
        if record is None:
            raise KeyError(frame_id)
        record.detections = []
        self.records[frame_id] = record.to_dict()
        write_yolo_label(self.labels_dir / f"{frame_id}.txt", record)
        with self.lock:
            self._persist()

    def export(self) -> dict:
        export_dir = self.root / "export"
        (export_dir / "labels").mkdir(parents=True, exist_ok=True)
        (export_dir / "payloads").mkdir(parents=True, exist_ok=True)
        (export_dir / "annotated").mkdir(parents=True, exist_ok=True)
        (export_dir / "coco").mkdir(parents=True, exist_ok=True)
        payloads = []
        records = []
        labeled = 0
        annotated = 0
        for frame_id in sorted(self.manifest):
            record = self.get_record(frame_id)
            records.append(record)
            write_yolo_label(export_dir / "labels" / f"{frame_id}.txt", record)
            payload = record.to_payload()
            (export_dir / "payloads" / f"{frame_id}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            payloads.append(payload)
            labeled += int((self.labels_dir / f"{frame_id}.txt").exists())
            rendered = render_annotated(record)
            if rendered is not None:
                (export_dir / "annotated" / f"{frame_id}.jpg").write_bytes(rendered)
                annotated += 1
        (export_dir / "payloads.jsonl").write_text(
            "\n".join(json.dumps(payload) for payload in payloads) + "\n", encoding="utf-8"
        )
        write_json_atomic(export_dir / "manifest.json", self.manifest)
        write_json_atomic(export_dir / "coco" / "annotations.json", build_coco(records))
        return {
            "export_dir": str(export_dir),
            "frames": len(payloads),
            "labeled": labeled,
            "detections": sum(len(payload["detections"]) for payload in payloads),
            "annotated": annotated,
            "coco": str(export_dir / "coco" / "annotations.json"),
        }


app = FastAPI(title="Road Anomaly Annotator")

_workspace: Workspace | None = None
_workspace_lock = threading.Lock()


def get_workspace() -> Workspace:
    global _workspace
    with _workspace_lock:
        if _workspace is None:
            _workspace = Workspace(Path(os.getenv("ANNOTATION_DIR", str(DEFAULT_WORKSPACE))))
        return _workspace


class DetectionsBody(BaseModel):
    detections: list[Detection] = []


class MetaBody(BaseModel):
    telemetry: dict = {}
    environment: dict = {}


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(UI_PATH)


@app.get("/api/classes")
async def api_classes():
    return {"classes": CLASS_NAMES, "severities": list(SEVERITY_LEVELS)}


@app.get("/api/frames")
async def api_frames(filter: str = "all"):
    return get_workspace().summaries(filter)


@app.get("/api/frames/{frame_id}")
async def api_get_frame(frame_id: str):
    record = get_workspace().get_record(frame_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown_frame")
    return record.to_dict()


@app.get("/api/frames/{frame_id}/image")
async def api_get_image(frame_id: str):
    workspace = get_workspace()
    entry = workspace.manifest.get(frame_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="unknown_frame")
    path = Path(entry["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="image_missing")
    return FileResponse(path)


@app.get("/api/frames/{frame_id}/annotated")
async def api_get_annotated(frame_id: str):
    record = get_workspace().get_record(frame_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown_frame")
    rendered = render_annotated(record)
    if rendered is None:
        raise HTTPException(status_code=404, detail="image_missing")
    return Response(content=rendered, media_type="image/jpeg")


@app.post("/api/frames/{frame_id}")
async def api_save_frame(frame_id: str, body: DetectionsBody):
    try:
        record = get_workspace().save_detections(frame_id, body.detections)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown_frame")
    return {"ok": True, "frame_id": frame_id, "detections": len(record.detections)}


@app.post("/api/frames/{frame_id}/meta")
async def api_save_meta(frame_id: str, body: MetaBody):
    try:
        get_workspace().save_meta(frame_id, body.telemetry, body.environment)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown_frame")
    return {"ok": True}


@app.delete("/api/frames/{frame_id}")
async def api_clear_frame(frame_id: str):
    try:
        get_workspace().clear_labels(frame_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown_frame")
    return {"ok": True}


@app.get("/api/export")
async def api_export():
    return get_workspace().export()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local annotation and reannotation server")
    parser.add_argument("--host", default=os.getenv("ANNOTATION_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ANNOTATION_PORT", "8765")))
    parser.add_argument("--dir", default=os.getenv("ANNOTATION_DIR", str(DEFAULT_WORKSPACE)))
    args = parser.parse_args()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    import uvicorn

    get_workspace()
    os.environ.setdefault("ANNOTATION_DIR", args.dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
