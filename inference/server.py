import logging
import os
import time
import sqlite3
import json
from datetime import datetime

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("yolo_pothole_detection")


API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is required for authentication.")


# --- Database Setup ---
INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("DB_PATH", os.path.join(INFERENCE_DIR, "potholes.db"))
UPLOADS_DIR = os.getenv("UPLOADS_DIR", os.path.join(INFERENCE_DIR, "uploads"))

os.makedirs(UPLOADS_DIR, exist_ok=True)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS potholes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT,
                confidence REAL,
                latitude REAL,
                longitude REAL,
                timestamp TEXT,
                image_path TEXT
            )
        """)
        conn.commit()

init_db()

def save_pothole(label, confidence, lat, lng, image_path):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO potholes (label, confidence, latitude, longitude, timestamp, image_path) VALUES (?, ?, ?, ?, ?, ?)",
            (label, confidence, lat, lng, datetime.utcnow().isoformat(), image_path),
        )
        conn.commit()

def get_all_potholes():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM potholes ORDER BY timestamp DESC")
        return [dict(row) for row in cursor.fetchall()]


def require_api_key(x_api_key: str | None = Header(default=None)):
    if x_api_key != API_KEY:
        logger.warning("Authentication failed for request.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_api_key"
        )


app = FastAPI(title="YOLO Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

DASHBOARD_PATH = os.path.join(INFERENCE_DIR, "dashboard", "index.html")


@app.get("/", include_in_schema=False)
async def dashboard():
    """Serve the self-contained dashboard from the Jetson API process."""
    return FileResponse(DASHBOARD_PATH)


@app.get("/potholes", dependencies=[Depends(require_api_key)])
async def get_potholes():
    try:
        return {"potholes": get_all_potholes()}
    except Exception as exc:
        logger.exception("Failed to fetch potholes: %s", exc)
        raise HTTPException(status_code=500, detail="database_error")


@app.get("/health")
async def health():
    status_path = os.getenv("STATUS_PATH", os.path.join(os.path.dirname(__file__), "local_status.json"))
    local_status = None
    try:
        with open(status_path, encoding="utf-8") as status_file:
            local_status = json.load(status_file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {"status": "ok", "local_runner": local_status}


@app.get("/local-status")
async def local_status():
    """Return status written by the optional all-on-Jetson camera runner."""
    status_path = os.getenv("STATUS_PATH", os.path.join(os.path.dirname(__file__), "local_status.json"))
    try:
        with open(status_path, encoding="utf-8") as status_file:
            return json.load(status_file)
    except FileNotFoundError:
        return {"status": "not_running"}
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=503, detail="local_status_unavailable") from exc
