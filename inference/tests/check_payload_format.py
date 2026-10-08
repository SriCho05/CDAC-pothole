import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotate.annotation_app import Workspace
from annotate.schema import Detection, Environment, FrameRecord, Telemetry

tmp = Path(tempfile.mkdtemp(prefix="payload_check_"))
ws = Workspace(tmp)
ws.manifest = {
    "TARAMANI_ROAD_02_09_2026": {
        "path": str(tmp / "frame.jpg"),
        "width": 1280,
        "height": 720,
        "timestamp_unix_us": 1787042200123456,
    }
}
ws.save_detections(
    "TARAMANI_ROAD_02_09_2026",
    [Detection(class_id=0, confidence=0.91, bbox_xywh=[480, 620, 140, 95], severity="level_2", estimated_depth_cm=6.5, estimated_area_sqcm=820.0)],
)
ws.save_meta(
    "TARAMANI_ROAD_02_09_2026",
    {"latitude": 12.985612, "longitude": 80.241289, "altitude_m": 14.2, "speed_kmh": 50, "heading_deg": 184.2, "imu_vibration_g": 0.18},
    {"lux": 42000, "surface": "asphalt_dry", "weather": "clear"},
)
print(json.dumps(ws.export(), indent=2))

saved = json.loads((tmp / "export" / "payloads" / "TARAMANI_ROAD_02_09_2026.json").read_text())
print("--- export/payloads/TARAMANI_ROAD_02_09_2026.json ---")
print(json.dumps(saved, indent=2))
print("--- export/payloads.jsonl (line 1) ---")
print((tmp / "export" / "payloads.jsonl").read_text().strip())
print("--- export/labels/TARAMANI_ROAD_02_09_2026.txt ---")
print((tmp / "export" / "labels" / "TARAMANI_ROAD_02_09_2026.txt").read_text())
shutil.rmtree(tmp, ignore_errors=True)
