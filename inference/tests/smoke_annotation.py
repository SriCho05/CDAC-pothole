import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

repo = Path(__file__).resolve().parents[2]
tmp = Path(tempfile.mkdtemp(prefix="smoke_"))
frames_dir = tmp / "frames"
frames_dir.mkdir()
cap = cv2.VideoCapture(str(repo / "20260912_102451.mp4"))
n = 0
while n < 3:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imwrite(str(frames_dir / f"frame_{n:04d}.jpg"), frame)
    n += 1
cap.release()
print("extracted", n, "frames to", tmp)

from annotate.preannotate import build_manifest_images, detect_frame, run
from annotate.annotation_app import Workspace
from annotate.schema import Detection

import argparse
import logging
logging.basicConfig(level=logging.INFO)

args = argparse.Namespace(
    source=str(frames_dir), out=str(tmp / "annotations"), confidence=0.25,
    image_size=640, device="cpu", pattern=None, every=10,
)
run(args)

from fastapi.testclient import TestClient
from annotate import annotation_app

client = TestClient(annotation_app.app)

workspace = Workspace(Path(args.out))
annotation_app._workspace = workspace

r = client.get("/api/classes")
assert r.status_code == 200 and len(r.json()["classes"]) >= 10, r.text

r = client.get("/api/frames")
assert r.status_code == 200, r.text
summary = r.json()
print("summary:", {k: summary[k] for k in ("total", "labeled", "detections")})
assert summary["total"] == n
frame_id = summary["frames"][0]["frame_id"]

r = client.get(f"/api/frames/{frame_id}")
assert r.status_code == 200, r.text
record = r.json()
assert record["frame_id"] == frame_id and record["detections"] is not None

r = client.get(f"/api/frames/{frame_id}/image")
assert r.status_code == 200 and r.headers["content-type"].startswith("image/")

saved = record["detections"][:1] or [{"class_id": 0, "confidence": 0.9, "bbox_xywh": [10, 10, 50, 60], "severity": "level_2"}]
r = client.post(f"/api/frames/{frame_id}", json={"detections": saved})
assert r.status_code == 200, r.text

r = client.post(
    f"/api/frames/{frame_id}/meta",
    json={"telemetry": {"latitude": 12.985612, "longitude": 80.241289, "speed_kmh": 50}, "environment": {"lux": 42000, "surface": "asphalt_dry", "weather": "clear"}},
)
assert r.status_code == 200, r.text

r = client.get(f"/api/frames/{frame_id}")
assert r.json()["telemetry"]["latitude"] == 12.985612
assert r.json()["environment"]["weather"] == "clear"
det = r.json()["detections"][0]
assert det["class_id"] == saved[0]["class_id"]

r = client.get("/api/export")
assert r.status_code == 200, r.text
export = r.json()
print("export:", export)
assert export["frames"] == n

payload = (Path(args.out) / "export" / "payloads" / f"{frame_id}.json").read_text()
import json
payload = json.loads(payload)
assert set(payload) == {"frame_id", "timestamp_unix_us", "telemetry", "environment", "detections"}
assert payload["telemetry"]["latitude"] == 12.985612
assert set(payload["detections"][0]) == {"class", "severity", "confidence", "bbox_rgb_xywh", "bbox_thermal_xywh", "estimated_depth_cm", "estimated_area_sqcm"}

label_line = (Path(args.out) / "export" / "labels" / f"{frame_id}.txt").read_text().strip().splitlines()[0]
parts = label_line.split()
assert len(parts) == 5 and 0 <= int(parts[0]) < 10

print("SMOKE TEST PASSED")
shutil.rmtree(tmp, ignore_errors=True)
