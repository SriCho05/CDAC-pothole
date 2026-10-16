"""Run inference, open the annotator UI, block until export, then proceed.

Usage:

    python -m annotate.pipeline \
        --source /path/to/video.mp4 \
        --out annotations \
        [--port 8765] [--no-browser] [--timeout 3600]

Flow:
  1. Run pre-annotation (inference over frames) to build the annotation workspace.
  2. Start the annotation FastAPI server.
  3. Open the browser to the annotator UI.
  4. Block until management clicks "Export dataset" in the UI.
  5. Stop the server and return success so the downstream pipeline (train/deploy)
     may move forward.

The annotator's "Export dataset" button writes a `.export_done` flag file and the
`/api/pipeline/status` endpoint reports it. This script polls that endpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

INFERENCE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE = INFERENCE_DIR / "annotations"

logger = logging.getLogger("annotate_pipeline")


def run_preannotate(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out)
    cmd = [
        sys.executable, "-m", "annotate.preannotate",
        "--source", args.source,
        "--out", str(out_dir),
        "--conf", str(args.conf),
        "--image-size", str(args.image_size),
        "--every", str(args.every),
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.pattern:
        cmd += ["--pattern", args.pattern]
    logger.info("Pre-annotating (inference) over %s -> %s", args.source, out_dir)
    subprocess.run(cmd, check=True, cwd=str(INFERENCE_DIR))
    return out_dir


def start_annotator(workspace: Path, host: str, port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["ANNOTATION_DIR"] = str(workspace)
    env["ANNOTATION_HOST"] = host
    env["ANNOTATION_PORT"] = str(port)
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "annotate.annotation_app",
            "--host", host, "--port", str(port), "--dir", str(workspace),
        ],
        env=env,
        cwd=str(INFERENCE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_for_server(port: int, timeout: float = 30.0) -> bool:
    url = f"http://127.0.0.1:{port}/api/classes"
    deadline = time.time() + timeout
    last_err = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    logger.info("Annotator server ready on http://127.0.0.1:%d/", port)
                    return True
        except Exception as exc:
            last_err = str(exc)
        time.sleep(0.5)
    logger.error("Annotator server did not become ready: %s", last_err)
    return False


def wait_for_export(port: int, timeout: float | None) -> bool:
    url = f"http://127.0.0.1:{port}/api/pipeline/status"
    deadline = time.time() + timeout if timeout else None
    while True:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("exported"):
                    logger.info("Export detected: %d frames, %d detections",
                                data.get("frames"), data.get("detections"))
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError) as exc:
            logger.debug("Polling status: %s", exc)
        except Exception as exc:  # noqa: BLE001 - transient poll errors are expected
            logger.debug("Polling status: %s", exc)
        if deadline and time.time() > deadline:
            logger.error("Timed out waiting for export")
            return False
        time.sleep(1.0)


def run_train(work: Path, args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, "-m", "annotate.train_pipeline",
        "--annotations", str(work),
        "--model", args.train_model,
        "--epochs", str(args.train_epochs),
        "--image-size", str(args.train_image_size),
        "--batch", str(args.train_batch),
        "--device", args.train_device,
        "--run-name", args.train_run_name,
    ]
    if args.train_tensorrt:
        cmd.append("--tensorrt")
    if args.train_int8:
        cmd.append("--int8")
    if args.train_calib_set:
        cmd += ["--calib-set", args.train_calib_set]
    if args.train_ignore_as_class:
        cmd.append("--ignore-as-class")
    logger.info("Starting training: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(INFERENCE_DIR)).returncode
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run inference, open annotator, block until export, then proceed (optionally train)."
    )
    parser.add_argument("--source", required=True, help="Image directory or video file to pre-annotate")
    parser.add_argument("--out", default=str(DEFAULT_WORKSPACE), help="Annotation workspace directory")
    parser.add_argument("--host", default="127.0.0.1", help="Annotator bind host")
    parser.add_argument("--port", type=int, default=8765, help="Annotator bind port")
    parser.add_argument("--conf", type=float, default=float(os.getenv("CONFIDENCE", "0.25")), help="Detection confidence")
    parser.add_argument("--image-size", type=int, default=int(os.getenv("IMAGE_SIZE", "640")), help="Inference image size")
    parser.add_argument("--every", type=int, default=10, help="Video: keep every Nth frame")
    parser.add_argument("--device", default=os.getenv("YOLO_DEVICE"), help="cuda:0 or cpu")
    parser.add_argument("--pattern", default=None, help="Comma-separated glob patterns, e.g. '*.jpg,*.png'")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window")
    parser.add_argument("--timeout", type=float, default=None, help="Max seconds to wait for export before aborting")
    parser.add_argument("--log-level", default="INFO")
    # Optional automatic training after export
    parser.add_argument("--train", action="store_true", help="Start training automatically after export")
    parser.add_argument("--train-model", default=str(INFERENCE_DIR / "best.pt"), help="Base model for training")
    parser.add_argument("--train-epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--train-image-size", type=int, default=int(os.getenv("IMAGE_SIZE", "640")), help="Training image size")
    parser.add_argument("--train-batch", type=int, default=int(os.getenv("BATCH_SIZE", "16")), help="Training batch size")
    parser.add_argument("--train-device", default=os.getenv("YOLO_DEVICE", "cuda:0"), help="Training device")
    parser.add_argument("--train-run-name", default="annotate_train", help="Training run name")
    parser.add_argument("--train-tensorrt", action="store_true", help="Build TensorRT engine after training")
    parser.add_argument("--train-int8", action="store_true", help="INT8 engine (requires --train-calib-set)")
    parser.add_argument("--train-calib-set", default=None, help="Calibration image path for INT8")
    parser.add_argument("--train-ignore-as-class", action="store_true", help="Remap ignore regions to background class")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s | %(levelname)s | %(message)s")

    # 1. Inference -> annotation workspace
    workspace = run_preannotate(args)

    # 2. Launch annotator server
    proc = start_annotator(workspace, args.host, args.port)
    try:
        if not wait_for_server(args.port):
            return 1

        # 3. Open browser
        url = f"http://{args.host}:{args.port}/"
        if not args.no_browser:
            logger.info("Opening annotator: %s", url)
            webbrowser.open(url)
        else:
            logger.info("Annotator UI: %s (browser disabled)", url)

        # 4. Block until management exports
        logger.info("Waiting for management to click 'Export dataset' in the annotator...")
        if not wait_for_export(args.port, args.timeout):
            return 1

        logger.info("Export complete. Pipeline may proceed.")

        # 5. Optional automatic training after export
        if args.train:
            rc = run_train(workspace, args)
            if rc != 0:
                logger.error("Training failed with exit code %d", rc)
                return rc
            logger.info("Training complete.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
