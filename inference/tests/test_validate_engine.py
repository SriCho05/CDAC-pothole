import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np

from annotate.validate_engine import box_iou, summarize, compare_detections


def test_box_iou_identical_boxes():
    box = [10, 20, 110, 120]
    assert box_iou(box, box) == 1.0


def test_box_iou_disjoint():
    assert box_iou([0, 0, 10, 10], [100, 100, 110, 110]) == 0.0


def test_box_iou_partial_overlap():
    iou = box_iou([0, 0, 100, 100], [50, 50, 150, 150])
    assert abs(iou - (50 * 50) / (100 * 100 + 100 * 100 - 50 * 50)) < 1e-6


def test_summarize_basic_stats():
    stats = summarize([10.0, 20.0, 30.0])
    assert stats["avg_inf_ms"] == 20.0
    assert stats["min_ms"] == 10.0
    assert stats["max_ms"] == 30.0
    assert stats["fps"] == 50.0


def test_summarize_percentiles():
    stats = summarize([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    assert stats["p50_ms"] == 5.5
    assert stats["min_ms"] == 1.0
    assert stats["max_ms"] == 10.0


def test_compare_detections_perfect_match():
    det = [{"class_id": 0, "confidence": 0.9, "bbox_xyxy": [10, 20, 110, 120]}]
    result = compare_detections([det], [det])
    assert result["matched"] == 1
    assert result["missed"] == 0
    assert result["false_positives"] == 0
    assert result["mean_iou"] == 1.0
    assert result["max_confidence_delta"] == 0.0


def test_compare_detections_miss_and_false_positive():
    pytorch = [[{"class_id": 0, "confidence": 0.9, "bbox_xyxy": [10, 10, 110, 110]}]]
    tensorrt = [[
        {"class_id": 0, "confidence": 0.88, "bbox_xyxy": [12, 12, 108, 108]},  # matched ~
        {"class_id": 0, "confidence": 0.7, "bbox_xyxy": [500, 500, 600, 600]},   # false positive
    ]]
    result = compare_detections(pytorch, tensorrt)
    assert result["matched"] == 1
    assert result["missed"] == 0
    assert result["false_positives"] == 1
    assert result["mean_iou"] > 0.9
    assert result["max_confidence_delta"] == 0.02


def test_compare_detections_class_mismatch():
    pytorch = [[{"class_id": 0, "confidence": 0.9, "bbox_xyxy": [10, 10, 110, 110]}]]
    tensorrt = [[{"class_id": 1, "confidence": 0.95, "bbox_xyxy": [10, 10, 110, 110]}]]
    result = compare_detections(pytorch, tensorrt)
    assert result["matched"] == 0
    assert result["missed"] == 1
    assert result["false_positives"] == 1
