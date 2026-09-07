"""Camera capture helpers for USB/V4L2 and Jetson CSI cameras."""

from __future__ import annotations

import os


def build_csi_pipeline(
    sensor_id: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    output_width: int = 640,
    output_height: int = 640,
) -> str:
    """Build an nvarguscamerasrc pipeline for a Jetson CSI camera."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1,format=NV12 ! "
        f"nvvidconv ! video/x-raw,format=BGRx,width={output_width},height={output_height} ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
    )


def open_camera(cv2, camera_type: str, device: str, width: int, height: int, fps: int):
    """Open a configured camera. ``cv2`` is injected to keep imports testable."""
    camera_type = camera_type.lower()
    if camera_type == "csi":
        source = build_csi_pipeline(
            sensor_id=int(os.getenv("CSI_SENSOR_ID", "0")),
            width=width,
            height=height,
            fps=fps,
            output_width=int(os.getenv("INFERENCE_WIDTH", "640")),
            output_height=int(os.getenv("INFERENCE_HEIGHT", "640")),
        )
        capture = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    elif camera_type == "usb":
        source = int(device) if str(device).isdigit() else device
        capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
    else:
        raise ValueError("CAMERA_TYPE must be 'usb' or 'csi'")

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {camera_type} camera: {device}")
    return capture