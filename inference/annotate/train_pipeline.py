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
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from .classes import CLASS_NAMES
except ImportError:  # Running directly as a script inside annotate/
    from classes import CLASS_NAMES

try:
    from .env_setup import ensure_cuda_libs
except ImportError:  # Running directly as a script inside annotate/
    from env_setup import ensure_cuda_libs

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


def _class_signature(label_path: Path) -> tuple[int, ...]:
    """Sorted tuple of class ids present in a YOLO label file."""
    classes = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            try:
                classes.add(int(parts[0]))
            except ValueError:
                continue
    return tuple(sorted(classes))


def prepare_yolo_dataset(
    work: Path,
    labels_dir: Path,
    include_ignore: bool,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Path, dict[str, str]]:
    """Build a YOLO-standard dataset tree with train/val/test splits.

    Ultralytics resolves a label file by replacing the ``/images/`` path
    component with ``/labels/``, so the training tree must use that exact
    layout - the flat ``export/`` layout is not loadable by YOLO.

    Training images are the ORIGINAL frames (``frames/``), never the
    ``export/annotated/`` renders: those have the detection boxes drawn
    on them, and training on them would teach the model to find boxes.

    Splits are stratified by the set of classes present in each frame so
    the class distribution stays balanced across train/val/test.
    """
    images_src = work / "frames"
    if not images_src.is_dir():
        images_src = work / "export" / "annotated"
        logger.warning("frames/ not found, falling back to export/annotated/ (boxes are drawn on these)")
    if not images_src.is_dir():
        raise SystemExit(f"No training images found under {work}")

    label_ids = {p.stem for p in labels_dir.glob("*.txt")}
    image_ids = {p.stem for p in images_src.glob("*.jpg")}
    shared = sorted(label_ids & image_ids)
    if not shared:
        raise SystemExit(f"No image/label pairs found in {images_src} + {labels_dir}")

    # Stratify: group frames by their class signature, then split each group.
    groups: dict[tuple[int, ...], list[str]] = {}
    for frame_id in shared:
        groups.setdefault(_class_signature(labels_dir / f"{frame_id}.txt"), []).append(frame_id)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for signature in sorted(groups):
        members = sorted(groups[signature])
        rng.shuffle(members)
        count = len(members)
        n_val = int(round(count * val_ratio)) if count >= 5 else 0
        n_test = int(round(count * test_ratio)) if count >= 10 else 0
        if count >= 3 and n_val == 0:
            n_val = 1
        if count >= 3 and n_test == 0 and n_val < count:
            n_test = 1
        n_val = min(n_val, max(count - 1, 0))
        n_test = min(n_test, max(count - 1 - n_val, 0))
        splits["val"].extend(members[:n_val])
        splits["test"].extend(members[n_val:n_val + n_test])
        splits["train"].extend(members[n_val + n_test:])

    # Guarantee every split is non-empty when the dataset allows it
    total = len(shared)
    if total >= 3:
        if not splits["val"]:
            splits["val"] = [splits["train"].pop()]
        if not splits["test"]:
            splits["test"] = [splits["train"].pop()]

    dataset_root = work / "yolo_dataset"
    for split, members in splits.items():
        image_dir = dataset_root / "images" / split
        label_out = dataset_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for stale in list(image_dir.glob("*.jpg")) + list(label_out.glob("*.txt")):
            stale.unlink()
        for frame_id in members:
            source_image = next(images_src.glob(f"{frame_id}.*"), None)
            if source_image is None:
                continue
            shutil.copy2(source_image, image_dir / f"{frame_id}{source_image.suffix}")
            shutil.copy2(labels_dir / f"{frame_id}.txt", label_out / f"{frame_id}.txt")

    logger.info(
        "YOLO dataset: %s (train=%d val=%d test=%d) from %d labeled frames",
        dataset_root, len(splits["train"]), len(splits["val"]), len(splits["test"]), total,
    )
    refs = {name: f"images/{name}" for name in ("train", "val", "test")}
    return dataset_root, refs


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
    else:
        label_dir = work / "export" / "labels"
        if not label_dir.exists():
            logger.error("No labels found. Run: python -m annotate.preannotate --source <frames> --out annotations")
            logger.error("then start the annotator, review, and click 'Export dataset'.")
            sys.exit(1)

    # Build the YOLO-standard tree (images/ + labels/, train/val/test) and
    # point the data.yaml at it.
    dataset_root, split_refs = prepare_yolo_dataset(
        work, label_dir, include_ignore,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed,
    )
    names = NAMES_LIST if include_ignore else NAMES_LIST[:10]
    nc = 11 if include_ignore else 10
    yaml_path.write_text(
        f"path: {dataset_root.resolve()}\n"
        f"train: {split_refs['train']}\n"
        f"val: {split_refs['val']}\n"
        f"test: {split_refs['test']}\n"
        f"nc: {nc}\n"
        f"names: {names}\n",
        encoding="utf-8",
    )
    logger.info("Wrote %s (nc=%d)", yaml_path, nc)

    model = YOLO(str(model_path))
    logger.info("Training with %s on device %s for %d epochs (amp=%s)", model_path, args.device, args.epochs, not args.no_amp)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        name=args.run_name,
        workers=args.workers,
        seed=args.seed,
        amp=not args.no_amp,
        plots=args.plots,
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
    """Build a TensorRT engine from the exported ONNX.

    INT8 goes through ``build_int8_engine.py`` (TensorRT Python API with an
    IInt8EntropyCalibrator2 over real images). ``trtexec`` has no
    ``--calib`` option, so an INT8 build cannot be done from the CLI.
    FP16 is built with ``trtexec --fp16``.
    """
    trt_dir = onnx_path.parent
    engine_name = f"{onnx_path.stem}_{'int8' if args.int8 else 'fp16'}.engine"
    engine_path = trt_dir / engine_name

    if args.int8:
        if not args.calib_set:
            logger.error("--int8 requires --calib-set pointing at the calibration images")
            sys.exit(1)
        cmd = [
            sys.executable, str(INFERENCE_DIR / "build_int8_engine.py"),
            "--onnx", str(onnx_path),
            "--calib-dir", str(args.calib_set),
            "--output", str(engine_path),
            "--cache", str(trt_dir / f"{onnx_path.stem}_calibration.cache"),
        ]
    else:
        cmd = [
            "trtexec",
            "--onnx=" + str(onnx_path),
            "--saveEngine=" + str(engine_path),
            "--fp16",
            "--workspace=2048",
        ]

    logger.info("Building TensorRT engine: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(INFERENCE_DIR), text=True, capture_output=True, timeout=3600)
    if result.returncode != 0:
        logger.error("TensorRT build failed:\n%s", result.stderr[-4000:])
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
    ensure_cuda_libs()  # re-exec with pip-installed CUDA libs on the loader path
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
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of frames held out for validation")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Fraction of frames held out for testing")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP (use FP32 instead of FP16 autocast)")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--plots", action="store_true", help="Save training plots (needs matplotlib 3D support)")
    parser.add_argument("--tensorrt", action="store_true", help="Build a TensorRT engine after ONNX export")
    parser.add_argument("--int8", action="store_true", help="INT8 engine (requires --calib-set)")
    parser.add_argument("--calib-set", default=None, help="Path to calibration images for INT8")
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s | %(levelname)s | %(message)s")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
