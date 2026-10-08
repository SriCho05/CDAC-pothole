import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import json

from annotate.classes import CLASS_IDS, CLASS_NAMES, SEVERITY_LEVELS, suggest_severity
from annotate.schema import (
    Detection,
    Environment,
    FrameRecord,
    Telemetry,
    build_coco,
    clamp_box,
    read_yolo_label,
    write_json_atomic,
    write_yolo_label,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def test_class_registry_is_bidirectional():
    assert len(CLASS_NAMES) >= 10
    assert set(CLASS_IDS.values()) == set(CLASS_NAMES.keys())
    assert CLASS_NAMES[0] == "pothole"
    assert CLASS_IDS["road_barrier"] in CLASS_NAMES
    assert set(SEVERITY_LEVELS) == {"level_1", "level_2", "level_3"}


def test_xyxy_xywh_conversion_roundtrip():
    box = [12.0, 34.0, 56.0, 78.0]
    assert xywh_to_xyxy(xyxy_to_xywh(box)) == box
    assert xyxy_to_xywh(xywh_to_xyxy(box)) == box


def test_detection_yolo_roundtrip(tmp_path):
    record = FrameRecord(
        frame_id="FRAME_0001",
        timestamp_unix_us=1787042200123456,
        image_width=1280,
        image_height=720,
        detections=[Detection(class_id=0, confidence=0.91, bbox_xywh=[480.0, 620.0, 140.0, 95.0], severity="level_2")],
    )
    label = tmp_path / "labels" / "FRAME_0001.txt"
    write_yolo_label(label, record)
    restored = read_yolo_label(label, 1280, 720)
    assert len(restored) == 1
    assert restored[0].class_id == 0
    assert abs(restored[0].confidence - 0.0) < 1e-9  # confidence lives in predictions.json, not the label file
    for actual, expected in zip(restored[0].bbox_xywh, record.detections[0].bbox_xywh):
        assert abs(actual - expected) < 1.0


def test_payload_matches_cdac_schema():
    record = FrameRecord(
        frame_id="TARAMANI_ROAD_02_09_2026",
        timestamp_unix_us=1787042200123456,
        image_width=1280,
        image_height=720,
        telemetry=Telemetry(latitude=12.985612, longitude=80.241289, altitude_m=14.2, speed_kmh=50, heading_deg=184.2, imu_vibration_g=0.18),
        environment=Environment(lux=42000, surface="asphalt_dry", weather="clear"),
        detections=[Detection(class_id=0, confidence=0.91, bbox_xywh=[480.0, 620.0, 140.0, 95.0], severity="level_2", estimated_depth_cm=6.5, estimated_area_sqcm=820.0)],
    )
    payload = record.to_payload()
    assert set(payload) == {"frame_id", "timestamp_unix_us", "telemetry", "environment", "detections"}
    assert payload["telemetry"] == {"latitude": 12.985612, "longitude": 80.241289, "altitude_m": 14.2, "speed_kmh": 50, "heading_deg": 184.2, "imu_vibration_g": 0.18}
    assert payload["environment"] == {"lux": 42000, "surface": "asphalt_dry", "weather": "clear"}
    detection = payload["detections"][0]
    assert set(detection) == {"class", "severity", "confidence", "bbox_rgb_xywh", "bbox_thermal_xywh", "estimated_depth_cm", "estimated_area_sqcm"}
    assert detection["class"] == "pothole"
    assert detection["bbox_rgb_xywh"] == [480.0, 620.0, 140.0, 95.0]
    assert detection["bbox_thermal_xywh"] is None
    assert detection["estimated_depth_cm"] == 6.5
    assert detection["estimated_area_sqcm"] == 820.0
    assert json.loads(json.dumps(payload)) == payload


def test_record_dict_roundtrip():
    record = FrameRecord(
        frame_id="FRAME_0002",
        timestamp_unix_us=1787042200999999,
        image_width=1920,
        image_height=1080,
        source_path="/data/frames/f2.jpg",
        telemetry=Telemetry(latitude=12.98, longitude=80.24),
        environment=Environment(lux=120, surface="asphalt_wet", weather="rainy"),
        detections=[
            Detection(class_id=4, confidence=0.77, bbox_xywh=[10.0, 20.0, 30.0, 40.0], severity="level_3", estimated_depth_cm=2.0),
            Detection(class_id=8, confidence=0.55, bbox_xywh=[100.0, 200.0, 300.0, 400.0]),
        ],
    )
    assert FrameRecord.from_dict(record.to_dict()).to_dict() == record.to_dict()


def test_suggest_severity_thresholds():
    assert suggest_severity(None) == "level_1"
    assert suggest_severity(0) == "level_1"
    assert suggest_severity(399.9) == "level_1"
    assert suggest_severity(400) == "level_2"
    assert suggest_severity(1499.9) == "level_2"
    assert suggest_severity(1500) == "level_3"


def test_clamp_box_to_image_bounds():
    assert clamp_box([-10, -5, 9999, 9999], 100, 100) == [0, 0, 100, 100]


def test_empty_label_writes_empty_file(tmp_path):
    label = tmp_path / "empty.txt"
    write_yolo_label(label, FrameRecord(frame_id="X", image_width=640, image_height=480))
    assert label.exists()
    assert label.read_text() == ""
    assert read_yolo_label(label, 640, 480) == []


def test_write_json_atomic_creates_parents(tmp_path):
    path = tmp_path / "nested" / "out.json"
    write_json_atomic(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert not (tmp_path / "nested" / "out.json.tmp").exists()


def test_export_writes_annotated_images(tmp_path):
    import cv2
    import numpy as np

    from annotate.annotation_app import Workspace

    frame_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(frame_path), np.full((200, 300, 3), 40, dtype=np.uint8))
    workspace = Workspace(tmp_path / "ws")
    workspace.manifest = {
        "F1": {"path": str(frame_path), "width": 300, "height": 200, "timestamp_unix_us": 1}
    }
    workspace.save_detections(
        "F1",
        [Detection(class_id=0, confidence=0.9, bbox_xywh=[10.0, 20.0, 50.0, 60.0], severity="level_2")],
    )
    result = workspace.export()
    assert result["annotated"] == 1
    annotated = tmp_path / "ws" / "export" / "annotated" / "F1.jpg"
    assert annotated.exists()
    rendered = cv2.imread(str(annotated))
    assert rendered is not None
    assert rendered.shape[:2] == (200, 300)


def test_build_coco_dataset():
    records = [
        FrameRecord(
            frame_id="FRAME_A",
            image_width=1280,
            image_height=720,
            source_path="/data/frames/FRAME_A.jpg",
            detections=[
                Detection(class_id=0, confidence=0.91, bbox_xywh=[480.0, 620.0, 140.0, 95.0], severity="level_2"),
                Detection(class_id=4, confidence=0.55, bbox_xywh=[10.0, 20.0, 30.0, 40.0]),
            ],
        ),
        FrameRecord(frame_id="FRAME_B", image_width=640, image_height=480, source_path="/data/frames/FRAME_B.png"),
    ]
    coco = build_coco(records)
    assert set(coco) == {"images", "annotations", "categories"}
    assert [image["file_name"] for image in coco["images"]] == ["FRAME_A.jpg", "FRAME_B.png"]
    assert coco["images"][0] == {"id": 0, "file_name": "FRAME_A.jpg", "width": 1280, "height": 720}
    assert coco["images"][1] == {"id": 1, "file_name": "FRAME_B.png", "width": 640, "height": 480}
    assert len(coco["annotations"]) == 2
    first = coco["annotations"][0]
    assert first["image_id"] == 0
    assert first["category_id"] == 1  # COCO categories are 1-based
    assert first["bbox"] == [480, 620, 140, 95]
    assert first["area"] == 140 * 95
    assert first["iscrowd"] == 0
    assert first["score"] == 0.91
    assert first["severity"] == "level_2"
    assert coco["annotations"][1]["category_id"] == 5  # debris (class 4) + offset
    assert len(coco["categories"]) == len(CLASS_NAMES)
    assert coco["categories"][0] == {"id": 1, "name": "pothole", "supercategory": "road_anomaly"}
    category_names = {category["name"] for category in coco["categories"]}
    assert category_names == set(CLASS_NAMES.values())
    assert json.loads(json.dumps(coco)) == coco
