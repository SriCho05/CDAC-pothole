#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/pothole-detection}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Warning: this installer is intended for Jetson aarch64; detected $(uname -m)."
fi

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-opencv v4l-utils gstreamer1.0-tools

sudo mkdir -p "$APP_DIR"
sudo cp -a . "$APP_DIR/inference"
sudo chown -R "$USER:$USER" "$APP_DIR"

cd "$APP_DIR/inference"
rm -rf .venv
$PYTHON_BIN -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install NVIDIA's JetPack-compatible torch/torchvision before this step.
# Do not replace it with generic ARM64/x86 CUDA wheels.
python -m pip install -r requirements-jetson.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created $APP_DIR/inference/.env; edit API_KEY, camera, and GNSS settings."
fi

sudo usermod -aG dialout "$USER" || true
sudo cp jetson/pothole-runner@.service /etc/systemd/system/
sudo cp jetson/pothole-api@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable "pothole-runner@$USER.service" "pothole-api@$USER.service"

echo "Installation complete. Log out/in for dialout permissions, then start:"
echo "  sudo systemctl start pothole-runner@$USER pothole-api@$USER"
