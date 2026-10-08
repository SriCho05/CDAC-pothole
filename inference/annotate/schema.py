"""Detection records, C-DAC JSON payloads, and YOLO label I/O."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .classes import CLASS_NAMES
except ImportError:  # Running directly as a script inside annotate/
    from classes import CLASS_NAMES


def _round_number(value: float) -> int | float:
    rounded = round(float(value), 2)
    return int(rounded) if rounded == int(rounded) else rounded


def build_coco(records: list["FrameRecord"], category_offset: int = 1) -> dict:
    """Build a COCO detection dataset from frame records.

    COCO category ids are 1-based by convention, so class ids are shifted
    by ``category_offset``. bbox is top-left [x, y, width, height] in
    pixels, matching the COCO detection spec. Extra keys (severity, score)
    are preserved; pycocotools tolerates them.
    """
    images = []
    annotations = []
    annotation_id = 1
    for image_id, record in enumerate(records):
        file_name = Path(record.source_path).name if record.source_path else f"{record.frame_id}.jpg"
        images.append({
            "id": image_id,
            "file_name": file_name,
            "width": record.image_width,
            "height": record.image_height,
        })
        for detection in record.detections:
            x, y, w, h = detection.bbox_xywh
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(detection.class_id) + category_offset,
                "bbox": [_round_number(value) for value in (x, y, w, h)],
                "area": _round_number(w * h),
                "iscrowd": 0,
                "score": round(float(detection.confidence), 4),
                "severity": detection.severity,
            })
            annotation_id += 1
    categories = [
        {
            "id": class_id + category_offset,
            "name": name,
            "supercategory": "road_anomaly",
        }
        for class_id, name in sorted(CLASS_NAMES.items())
    ]
    return {"images": images, "annotations": annotations, "categories": categories}


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def clamp_box(box: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = box
    x = min(max(0.0, x), float(width))
    y = min(max(0.0, y), float(height))
    w = min(max(0.0, w), float(width) - x)
    h = min(max(0.0, h), float(height) - y)
    return [x, y, w, h]


@dataclass
class Detection:
    class_id: int
    confidence: float = 0.0
    bbox_xywh: list[float] = field(default_factory=list)
    severity: str = "level_1"
    estimated_depth_cm: float | None = None
    estimated_area_sqcm: float | None = None

    def __post_init__(self) -> None:
        self.bbox_xywh = [float(value) for value in self.bbox_xywh]

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, f"class_{self.class_id}")

    @classmethod
    def from_xyxy(
        cls,
        class_id: int,
        xyxy: list[float],
        confidence: float = 0.0,
        severity: str = "level_1",
        estimated_depth_cm: float | None = None,
        estimated_area_sqcm: float | None = None,
    ) -> "Detection":
        return cls(
            class_id=class_id,
            confidence=confidence,
            bbox_xywh=xyxy_to_xywh(xyxy),
            severity=severity,
            estimated_depth_cm=estimated_depth_cm,
            estimated_area_sqcm=estimated_area_sqcm,
        )

    @classmethod
    def from_yolo(
        cls,
        class_id: int,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        image_width: int,
        image_height: int,
        confidence: float = 0.0,
    ) -> "Detection":
        xyxy = [
            (center_x - width / 2) * image_width,
            (center_y - height / 2) * image_height,
            (center_x + width / 2) * image_width,
            (center_y + height / 2) * image_height,
        ]
        return cls.from_xyxy(class_id, xyxy, confidence=confidence)

    def to_yolo(self, image_width: int, image_height: int) -> tuple[int, float, float, float, float]:
        x, y, w, h = self.bbox_xywh
        return (
            int(self.class_id),
            (x + w / 2) / image_width,
            (y + h / 2) / image_height,
            w / image_width,
            h / image_height,
        )

    def to_payload(self) -> dict:
        x, y, w, h = self.bbox_xywh
        return {
            "class": self.class_name,
            "severity": self.severity,
            "confidence": round(float(self.confidence), 4),
            "bbox_rgb_xywh": [_round_number(value) for value in (x, y, w, h)],
            "bbox_thermal_xywh": None,
            "estimated_depth_cm": self.estimated_depth_cm,
            "estimated_area_sqcm": self.estimated_area_sqcm,
        }

    def to_dict(self) -> dict:
        return {
            "class_id": int(self.class_id),
            "confidence": float(self.confidence),
            "bbox_xywh": self.bbox_xywh,
            "severity": self.severity,
            "estimated_depth_cm": self.estimated_depth_cm,
            "estimated_area_sqcm": self.estimated_area_sqcm,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Detection":
        return cls(
            class_id=int(data["class_id"]),
            confidence=float(data.get("confidence", 0.0)),
            bbox_xywh=list(data.get("bbox_xywh", [])),
            severity=data.get("severity", "level_1"),
            estimated_depth_cm=data.get("estimated_depth_cm"),
            estimated_area_sqcm=data.get("estimated_area_sqcm"),
        )


@dataclass
class Telemetry:
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    speed_kmh: float | None = None
    heading_deg: float | None = None
    imu_vibration_g: float | None = None

    FIELDS: tuple[str, ...] = (
        "latitude",
        "longitude",
        "altitude_m",
        "speed_kmh",
        "heading_deg",
        "imu_vibration_g",
    )

    def to_payload(self) -> dict:
        return {key: getattr(self, key) for key in self.FIELDS}

    def update(self, data: dict) -> None:
        for key in self.FIELDS:
            if key in data:
                setattr(self, key, data[key])

    @classmethod
    def from_dict(cls, data: dict) -> "Telemetry":
        telemetry = cls()
        telemetry.update(data or {})
        return telemetry


@dataclass
class Environment:
    lux: float | None = None
    surface: str | None = None
    weather: str | None = None

    FIELDS: tuple[str, ...] = ("lux", "surface", "weather")

    def to_payload(self) -> dict:
        return {key: getattr(self, key) for key in self.FIELDS}

    def update(self, data: dict) -> None:
        for key in self.FIELDS:
            if key in data:
                setattr(self, key, data[key])

    @classmethod
    def from_dict(cls, data: dict) -> "Environment":
        environment = cls()
        environment.update(data or {})
        return environment


@dataclass
class FrameRecord:
    frame_id: str
    timestamp_unix_us: int = 0
    image_width: int = 0
    image_height: int = 0
    source_path: str | None = None
    telemetry: Telemetry = field(default_factory=Telemetry)
    environment: Environment = field(default_factory=Environment)
    detections: list[Detection] = field(default_factory=list)

    def to_payload(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp_unix_us": int(self.timestamp_unix_us),
            "telemetry": self.telemetry.to_payload(),
            "environment": self.environment.to_payload(),
            "detections": [detection.to_payload() for detection in self.detections],
        }

    def to_dict(self) -> dict:
        data = self.to_payload()
        data["image_width"] = self.image_width
        data["image_height"] = self.image_height
        data["source_path"] = self.source_path
        data["detections"] = [detection.to_dict() for detection in self.detections]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "FrameRecord":
        return cls(
            frame_id=data["frame_id"],
            timestamp_unix_us=int(data.get("timestamp_unix_us", 0)),
            image_width=int(data.get("image_width", 0)),
            image_height=int(data.get("image_height", 0)),
            source_path=data.get("source_path"),
            telemetry=Telemetry.from_dict(data.get("telemetry")),
            environment=Environment.from_dict(data.get("environment")),
            detections=[Detection.from_dict(item) for item in data.get("detections", [])],
        )


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_yolo_label(path: Path, record: FrameRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for detection in record.detections:
        class_id, center_x, center_y, width, height = detection.to_yolo(
            record.image_width, record.image_height
        )
        lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    temporary.replace(path)


def read_yolo_label(path: Path, image_width: int, image_height: int) -> list[Detection]:
    if not path.exists():
        return []
    detections = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        detections.append(
            Detection.from_yolo(
                int(parts[0]),
                *map(float, parts[1:]),
                image_width=image_width,
                image_height=image_height,
            )
        )
    return detections
