"""Compare a TensorRT engine against its PyTorch source and ground-truth labels.

For every labelled frame this script runs three things:

  1. the PyTorch model (FP32 reference),
  2. the TensorRT engine (INT8/FP16 deployment artefact),
  3. the hand-corrected YOLO labels (ground truth).

It reports two numbers that matter for deployment:

  * engine-vs-pytorch agreement - greedy IoU matching of the two detection
    sets, so INT8 quantization loss is visible as a confidence/score drift
    rather than as missing boxes,
  * mAP50 / precision / recall of each model against the ground truth.

Usage:
    python inference/validate_engine_parity.py \\
        --model inference/retrained/best_retrained.pt \\
        --engine inference/retrained/best_retrained_int8.engine \\
        --frames inference/annotations/frames \\
        --labels inference/annotations/labels \\
        --output inference/retrained/parity_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

INFERENCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INFERENCE_DIR))

from annotate.classes import CLASS_NAMES  # noqa: E402

IMAGE_SIZE = 640
IOU_MATCH = 0.5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def preprocess(frame: np.ndarray) -> np.ndarray:
    """BGR frame -> normalised NCHW float32 tensor (Ultralytics layout)."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(np.expand_dims(img, axis=0))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a[:, None, :]
    b = b[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_counts(det_a: list[dict], det_b: list[dict], iou_thresh: float = IOU_MATCH):
    """Greedy IoU matching. Returns (matched, n_a, n_b)."""
    if not det_a or not det_b:
        return 0, len(det_a), len(det_b)
    boxes_a = np.array([d["bbox"] for d in det_a], dtype=np.float32)
    boxes_b = np.array([d["bbox"] for d in det_b], dtype=np.float32)
    ious = iou_matrix(boxes_a, boxes_b)
    used_b = set()
    matched = 0
    order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
    for i, j in order:
        if ious[i, j] < iou_thresh:
            break
        if i in used_b:
            continue
        used_b.add(i)
        matched += 1
    return matched, len(det_a), len(det_b)


def load_labels(label_path: Path, width: int, height: int) -> list[dict]:
    """Read a YOLO label file into xyxy detections in frame pixel space."""
    if not label_path.exists():
        return []
    dets = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:])
        x1 = (cx - w / 2) * width
        y1 = (cy - h / 2) * height
        x2 = (cx + w / 2) * width
        y2 = (cy + h / 2) * height
        dets.append({"bbox": [x1, y1, x2, y2], "confidence": 1.0, "class_id": class_id})
    return dets


def average_precision(detections: list[dict], ground_truth: list[dict]) -> tuple[float, float, float]:
    """Single-class mAP50 / precision / recall with greedy IoU matching.

    ``detections`` is a flat list over all frames; each entry carries
    ``frame_id``. ``ground_truth`` maps frame_id -> list of boxes.
    """
    if not detections:
        return 0.0, 0.0, 0.0
    by_frame: dict[str, list[dict]] = {}
    for det in detections:
        by_frame.setdefault(det["frame_id"], []).append(det)
    for dets in by_frame.values():
        dets.sort(key=lambda d: -d["confidence"])

    total_gt = sum(len(v) for v in ground_truth.values())
    matched = {fid: [False] * len(v) for fid, v in ground_truth.items()}
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    cursor = 0
    for fid, dets in by_frame.items():
        gt = ground_truth.get(fid, [])
        gt_boxes = np.array([g["bbox"] for g in gt], dtype=np.float32) if gt else np.zeros((0, 4))
        for det in dets:
            idx = cursor
            cursor += 1
            if not gt:
                fp[idx] = 1
                continue
            box = np.array([det["bbox"]], dtype=np.float32)
            ious = iou_matrix(box, gt_boxes)[0]
            order = np.argsort(-ious)
            for g_idx in order:
                if ious[g_idx] < IOU_MATCH:
                    break
                if matched[fid][g_idx]:
                    continue
                if gt[g_idx]["class_id"] != det["class_id"]:
                    continue
                matched[fid][g_idx] = True
                tp[idx] = 1
                break
            else:
                fp[idx] = 1

    if total_gt == 0:
        return 0.0, 0.0, 0.0
    order = np.argsort(-np.array([d["confidence"] for d in detections]))
    tp = tp[order]
    fp = fp[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = float(np.sum((recall[1:] - recall[:-1]) * precision[1:])) if len(recall) > 1 else 0.0
    precision = float(precision[-1]) if len(precision) else 0.0
    recall = float(recall[-1]) if len(recall) else 0.0
    return ap, precision, recall


# ---------------------------------------------------------------------------
# engine wrapper
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, engine_path: Path):
        logger = trt.Logger(trt.Logger.WARNING)
        # PyTorch and pycuda each create their own primary context. The
        # engine must run with the pycuda context pushed, otherwise
        # enqueueV3 fails with "Cuda Runtime (invalid resource handle)"
        # as soon as torch has made its context current.
        self.context = cuda.Device(0).make_context()
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.out_shape = tuple(self.ctx.get_tensor_shape("output0"))
        self.num_classes = self.out_shape[1] - 4
        self.d_in = cuda.mem_alloc(trt.volume((1, 3, IMAGE_SIZE, IMAGE_SIZE)) * 4)
        self.d_out = cuda.mem_alloc(trt.volume(self.out_shape) * 4)
        self.ctx.set_tensor_address("images", int(self.d_in))
        self.ctx.set_tensor_address("output0", int(self.d_out))
        self.stream = cuda.Stream()
        self.host_out = np.empty(self.out_shape, dtype=np.float32)

    def detect(self, frame, conf: float, nms: float) -> list[dict]:
        height, width = frame.shape[:2]
        self.context.push()
        try:
            cuda.memcpy_htod_async(self.d_in, preprocess(frame), self.stream)
            self.ctx.execute_async_v3(stream_handle=int(self.stream.handle))
            cuda.memcpy_dtoh_async(self.host_out, self.d_out, self.stream)
            self.stream.synchronize()
        finally:
            self.context.pop()

        pred = np.transpose(self.host_out[0], (1, 0))  # (anchors, 4 + nc)
        cls_scores = pred[:, 4:]
        if self.num_classes <= 1:
            class_ids = np.zeros(len(pred), dtype=np.int32)
            scores = cls_scores[:, 0]
        else:
            class_ids = np.argmax(cls_scores, axis=1)
            scores = cls_scores[np.arange(len(class_ids)), class_ids]
        mask = scores >= conf
        if not np.any(mask):
            return []
        sel = pred[mask]
        scores = scores[mask].astype(np.float32)
        class_ids = class_ids[mask]
        cx, cy, bw, bh = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
        sx, sy = width / IMAGE_SIZE, height / IMAGE_SIZE
        boxes = np.column_stack([
            (cx - bw / 2) * sx, (cy - bh / 2) * sy,
            (cx + bw / 2) * sx, (cy + bh / 2) * sy,
        ]).astype(np.float32)
        offset = class_ids.astype(np.float32) * 7680.0
        nms_input = boxes.copy()
        nms_input[:, 0] += offset
        nms_input[:, 1] += offset
        idx = cv2.dnn.NMSBoxes(nms_input.tolist(), scores.tolist(), conf, nms)
        out = []
        for i in np.array(idx).flatten():
            class_id = int(class_ids[i])
            out.append({
                "bbox": [float(boxes[i][0]), float(boxes[i][1]),
                         float(boxes[i][2]), float(boxes[i][3])],
                "confidence": float(scores[i]),
                "class_id": class_id,
                "label": CLASS_NAMES.get(class_id, f"class_{class_id}"),
            })
        return out


    def close(self) -> None:
        """Free the device buffers and pop the CUDA context."""
        try:
            self.d_in.free()
            self.d_out.free()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.context.pop()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="TensorRT vs PyTorch vs ground-truth parity check")
    parser.add_argument("--model", required=True, help="PyTorch source model (.pt)")
    parser.add_argument("--engine", required=True, help="TensorRT engine (.engine)")
    parser.add_argument("--frames", default=str(INFERENCE_DIR / "annotations" / "frames"))
    parser.add_argument("--labels", default=str(INFERENCE_DIR / "annotations" / "labels"))
    parser.add_argument("--output", default=str(INFERENCE_DIR / "engine_parity_report.json"))
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--nms", type=float, default=0.30)
    parser.add_argument("--max-frames", type=int, default=70)
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep the confidence threshold and report mAP50 for each")
    args = parser.parse_args()

    from ultralytics import YOLO

    frames_dir = Path(args.frames)
    labels_dir = Path(args.labels)
    frame_files = sorted(p for p in frames_dir.glob("*.jpg"))
    frame_files = frame_files[: args.max_frames]
    if not frame_files:
        raise SystemExit(f"No frames in {frames_dir}")

    model = YOLO(args.model)
    engine = Engine(Path(args.engine))
    print(f"Engine classes: {engine.num_classes}")

    frame_data: list[tuple[str, np.ndarray, int, int]] = []
    for path in frame_files:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        frame_data.append((path.stem, frame, width, height))

    # PyTorch and pycuda each own a primary CUDA context. Alternating
    # between them inside one loop breaks cuDNN (STREAM_MISMATCH) and the
    # engine (invalid resource handle), so each backend runs in its own
    # pass over the frames.
    pt_dets: list[dict] = []
    for frame_id, frame, _w, _h in frame_data:
        p_result = model.predict(frame, imgsz=IMAGE_SIZE, conf=args.confidence,
                                  iou=args.nms, device="cuda:0", verbose=False)[0]
        for box in p_result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls)
            pt_dets.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf),
                "class_id": class_id,
                "label": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                "frame_id": frame_id,
            })
    print(f"PyTorch pass done: {len(pt_dets)} detections")

    trt_dets: list[dict] = []
    for frame_id, frame, _w, _h in frame_data:
        for d in engine.detect(frame, args.confidence, args.nms):
            trt_dets.append({**d, "frame_id": frame_id})
    print(f"TensorRT pass done: {len(trt_dets)} detections")

    ground_truth: dict[str, list[dict]] = {}
    for frame_id, _frame, width, height in frame_data:
        ground_truth[frame_id] = load_labels(labels_dir / f"{frame_id}.txt", width, height)

    matched_total = 0
    trt_total = 0
    pt_total = 0
    for frame_id, _frame, _w, _h in frame_data:
        t = [d for d in trt_dets if d["frame_id"] == frame_id]
        p = [d for d in pt_dets if d["frame_id"] == frame_id]
        matched, n_trt, n_pt = match_counts(t, p)
        matched_total += matched
        trt_total += n_trt
        pt_total += n_pt

    trt_ap, trt_p, trt_r = average_precision(trt_dets, ground_truth)
    pt_ap, pt_p, pt_r = average_precision(pt_dets, ground_truth)

    sweep: list[dict] = []
    if args.sweep:
        print("\nConfidence threshold sweep (engine vs ground truth):")
        for threshold in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10,
                          0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
            hits = []
            for frame_id, frame, _w, _h in frame_data:
                for d in engine.detect(frame, threshold, args.nms):
                    hits.append({**d, "frame_id": frame_id})
            ap, prec, rec = average_precision(hits, ground_truth)
            sweep.append({
                "confidence": threshold,
                "detections": len(hits),
                "mAP50": round(ap, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
            })
            print(f"  conf {threshold:>4.2f} | det {len(hits):>4} | "
                  f"mAP50 {ap:.4f} | P {prec:.4f} | R {rec:.4f}")

    report = {
        "model": args.model,
        "engine": args.engine,
        "engine_num_classes": engine.num_classes,
        "frames_tested": len(ground_truth),
        "confidence_threshold": args.confidence,
        "iou_threshold": IOU_MATCH,
        "ground_truth_boxes": sum(len(v) for v in ground_truth.values()),
        "engine_vs_pytorch": {
            "matched_boxes": matched_total,
            "engine_detections": trt_total,
            "pytorch_detections": pt_total,
            "agreement_precision": round(matched_total / trt_total, 4) if trt_total else 0.0,
            "agreement_recall": round(matched_total / pt_total, 4) if pt_total else 0.0,
            "note": "greedy IoU>=0.5 match between engine and PyTorch detections",
        },
        "engine_vs_ground_truth": {
            "mAP50": round(trt_ap, 4),
            "precision": round(trt_p, 4),
            "recall": round(trt_r, 4),
        },
        "pytorch_vs_ground_truth": {
            "mAP50": round(pt_ap, 4),
            "precision": round(pt_p, 4),
            "recall": round(pt_r, 4),
        },
    }
    if sweep:
        report["threshold_sweep"] = sweep
    engine.close()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
