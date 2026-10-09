"""Train pipeline: export-corrected dataset -> YOLO fine-tune -> ONNX -> TensorRT engine.

Runs the full annotate->retrain->deploy loop on the Jetson:

    python -m annotate.train_pipeline \\
        --annotations ../annotations \\
        --model best.pt (or yolov8n.pt) \\
        [--tensorrt]  # build the .engine too

Produces: runs/annotate_train/ (YOLO), best_retrained.pt (ONNX), best_int8.engine / best_fp16.engine
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from .classes import CLASS_NAMES
except ImportError:  # Running directly as a script inside annotate/
    from classes import CLASS_NAMES

logger = logging.getLogger("annotate_train")

INFERENCE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE = INFERENCE_DIR / "annotations"
DEFAULT_MODEL = INFERENCE_DIR / "best.pt"
NAMES_LIST = ["pothole", "crack_transverse", "crack_longitudinal", "rutting", "debris",
              "waterlogging", "missing_lane_marking", "edge_break", "manhole", "road_barrier", "background"]


def write_yaml(path: Path, work: Path, labels_subdir: str, include_ignore: bool) -> None:
    names = NAMES_LIST if include_ignore else NAMES_LIST[:10]
    nc = 11 if include_ignore else 10
    content = (
        f"path: {work.resolve()}\n"
        f"train: {labels_subdir}\n"
        f"val: {labels_subdir}\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote %s (nc=%d)", path, nc)


def merge_ignore(work: Path) -> Path:
    """Write labels_merged/ with ignore regions remapped to class 10."""
    positive = work / "export" / "labels"
    ignore_dir = work / "export" / "labels_ignore"
    merged = work / "export" / "labels_merged"
    merged.mkdir(parents=True, exist_ok=True)
    for label_file in positive.glob("*.txt"):
        target = merged / label_file.name
        lines = label_file.read_text().splitlines()
        ignore_file = ignore_dir / label_file.name
        if ignore_file.exists():
            for line in ignore_file.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    parts[0] = "10"
                    lines.append(" ".join(parts))
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return merged


def run_yolo_train(args: argparse.Namespace, work: Path) -> Path:
    from ultralytics import YOLO

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        sys.exit(1)

    yaml_path = work / "data.yaml"
    include_ignore = args.ignore_as_class

    if include_ignore:
        label_dir = merge_ignore(work)
        labels_ref = "export/labels_merged"
    else:
        label_dir = work / "export" / "labels"
        if not label_dir.exists():
            logger.error("No labels found. Run: python -m annotate.preannotate --source <frames> --out annotations")
            logger.error("then start the annotator, review, and click 'Export dataset'.")
            sys.exit(1)
        labels_ref = "export/labels"
    write_yaml(yaml_path, work, labels_ref, include_ignore)

    model = YOLO(str(model_path))
    logger.info("Training with %s on device %s for %d epochs", model_path, args.device, args.epochs)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        name=args.run_name,
        workers=2,
        seed=args.seed,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.exists():
        best = Path("runs") / args.run_name / "weights" / "best.pt"
    if not best.exists():
        logger.error("Could not locate trained best.pt under %s", model.trainer.save_dir)
        sys.exit(1)
    logger.info("Trained weights: %s", best)
    return best


def export_onnx(best: Path, device: str) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(best))
    onnx_path = best.parent / f"{best.stem}.onnx"
    model.export(format="onnx", opset=12, device=device)
    logger.info("Exported ONNX: %s", onnx_path)
    return onnx_path


def build_trt_engine(onnx_path: Path, args: argparse.Namespace) -> Path:
    trt_dir = onnx_path.parent
    engine_name = f"{onnx_path.stem}_{'int8' if args.int8 else 'fp16'}.engine"
    engine_path = trt_dir / engine_name

    trt_cmd = [
        "trtexec",
        "--onnx=" + str(onnx_path),
        "--saveEngine=" + str(engine_path),
        "--workspace=2048",
    ]
    if args.int8:
        trt_cmd.extend(["--int8", f"--calib={args.calib_set or '[]'}"])
    else:
        trt_cmd.append("--fp16")

    logger.info("Building TensorRT engine: %s", " ".join(trt_cmd))
    result = subprocess.run(trt_cmd, cwd=str(INFERENCE_DIR), text=True, capture_output=True, timeout=1800)
    if result.returncode != 0:
        logger.error("TensorRT build failed:\n%s", result.stderr)
        sys.exit(1)
    logger.info("TensorRT engine: %s", engine_path)
    return engine_path


def run(args: argparse.Namespace) -> None:
    work = Path(args.annotations)
    started = time.perf_counter()

    # Ensure dataset is exported
    if not (work / "export" / "manifest.json").exists():
        from .annotation_app import Workspace

        workspace = Workspace(work)
        export_result = workspace.export()
        logger.info("Exported annotation workspace: %s", export_result)
        if export_result["detections"] == 0:
            logger.error("No detections found. Label frames in the annotator UI first.")
            sys.exit(1)

    best = run_yolo_train(args, work)

    # Deploy: copy best.pt next to the existing model as best_retrained.pt
    deploy_model = INFERENCE_DIR / "best_retrained.pt"
    shutil.copy2(best, deploy_model)
    logger.info("Deployed retrained model: %s", deploy_model)

    onnx_path = export_onnx(best, args.device)

    if args.tensorrt:
        engine_path = build_trt_engine(onnx_path, args)
        logger.info("TensorRT engine ready at %s", engine_path)
        logger.info("Set MODEL_PATH=%s in inference/.env to use the engine", engine_path)
    else:
        logger.info("Skipping TensorRT build. Add --tensorrt to build the engine.")

    elapsed = time.perf_counter() - started
    logger.info("Pipeline complete in %.1f seconds", elapsed)
    logger.info("To deploy: cp %s to best.pt on the Jetson, then: sudo systemctl restart pothole-runner@$USER pothole-api@$USER", deploy_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full train-and-deploy pipeline from annotated dataset")
    parser.add_argument("--annotations", default=str(DEFAULT_WORKSPACE), help="Annotation workspace with exported dataset")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Base model (best.pt or yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=int(os.getenv("IMAGE_SIZE", "640")))
    parser.add_argument("--batch", type=int, default=int(os.getenv("BATCH_SIZE", "16")))
    parser.add_argument("--device", default=os.getenv("YOLO_DEVICE", "cuda:0"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="annotate_train")
    parser.add_argument("--ignore-as-class", action="store_true", help="Remap ignore regions to class 10 (background)")
    parser.add_argument("--tensorrt", action="store_true", help="Build a TensorRT engine after ONNX export")
    parser.add_argument("--int8", action="store_true", help="INT8 engine (requires --calib-set)")
    parser.add_argument("--calib-set", default=None, help="Path to calibration images for INT8")
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s | %(levelname)s | %(message)s")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
