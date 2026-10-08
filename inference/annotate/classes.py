"""Anomaly class registry and severity levels for the CDAC road-survey schema."""

from __future__ import annotations

CLASS_NAMES: dict[int, str] = {
    0: "pothole",
    1: "crack_transverse",
    2: "crack_longitudinal",
    3: "rutting",
    4: "debris",
    5: "waterlogging",
    6: "missing_lane_marking",
    7: "edge_break",
    8: "manhole",
    9: "road_barrier",
}

CLASS_IDS: dict[str, int] = {name: class_id for class_id, name in CLASS_NAMES.items()}

SEVERITY_LEVELS: tuple[str, ...] = ("level_1", "level_2", "level_3")

SEVERITY_AREA_THRESHOLDS_CM2: tuple[float, float] = (400.0, 1500.0)


def class_id_for(name: str) -> int | None:
    return CLASS_IDS.get(name)


def class_name_for(class_id: int) -> str:
    return CLASS_NAMES.get(class_id, f"class_{class_id}")


def suggest_severity(area_sqcm: float | None) -> str:
    if area_sqcm is None:
        return "level_1"
    minor, moderate = SEVERITY_AREA_THRESHOLDS_CM2
    if area_sqcm < minor:
        return "level_1"
    if area_sqcm < moderate:
        return "level_2"
    return "level_3"
