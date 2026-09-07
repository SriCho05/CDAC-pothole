# Orion Nano deployment

This is the complete Jetson Orin Nano runtime. It runs camera capture, YOLO inference, GNSS reading, event storage, API, and dashboard on the Jetson.

## Start

From `inference/`:

```bash
export API_KEY=your-local-key
export MODEL_PATH=$PWD/best.pt
export CAMERA_TYPE=csi
export GNSS_PORT=/dev/ttyUSB0
python3 local_runner.py --camera-type csi --no-display
```

In a second terminal:

```bash
export API_KEY=your-local-key
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open this from any device on the same network:

```text
http://JETSON_IP:8000/
```

Enter the same `API_KEY` in the dashboard. The page is served directly by FastAPI; no Node.js, React, laptop, phone, or separate dashboard server is required.

## Dashboard features

- Runner status and camera mode
- Live inference FPS and frame latency
- GNSS fix state
- Latest annotated camera frame
- Saved pothole events with confidence and coordinates
- Interactive GIS map with one marker per GPS-tagged pothole
- Links to stored event images

The map uses Leaflet in the browser and OpenStreetMap tiles. The Jetson API and inference remain local; only map tiles require internet access. Without internet, the dashboard still shows coordinates and events, but the map background may be blank.

The dashboard reads public `/local-status` and protected `/potholes`. The API also serves `/uploads/latest.jpg` and saved event images.

## Services

Use `install_jetson.sh` to install the Python environment and systemd templates, then start:

```bash
sudo systemctl start pothole-runner@$USER pothole-api@$USER
```

The dashboard is available while `pothole-api` is running. The runner must also be running for live status and frames.
