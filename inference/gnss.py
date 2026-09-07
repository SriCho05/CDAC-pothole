"""Small dependency-free NMEA reader for Jetson USB/UART GNSS receivers."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Position:
    latitude: float
    longitude: float
    timestamp: str


def _coordinate(value: str, direction: str) -> float | None:
    if not value or not direction:
        return None
    try:
        degrees = int(float(value) // 100)
        minutes = float(value) - degrees * 100
        result = degrees + minutes / 60
        return -result if direction in ("S", "W") else result
    except (TypeError, ValueError):
        return None


def parse_nmea(sentence: str) -> Position | None:
    """Parse GGA or RMC NMEA sentences, returning None for invalid/no-fix data."""
    sentence = sentence.strip()
    if not sentence.startswith("$"):
        return None
    body = sentence[1:].split("*", 1)[0]
    fields = body.split(",")
    kind = fields[0][-3:]
    if kind == "GGA" and len(fields) >= 7 and fields[6] not in ("", "0"):
        lat = _coordinate(fields[2], fields[3])
        lon = _coordinate(fields[4], fields[5])
        if lat is not None and lon is not None:
            return Position(lat, lon, fields[1])
    if kind == "RMC" and len(fields) >= 7 and fields[2] == "A":
        lat = _coordinate(fields[3], fields[4])
        lon = _coordinate(fields[5], fields[6])
        if lat is not None and lon is not None:
            return Position(lat, lon, fields[1])
    return None


class GnssReader:
    """Background serial reader; unavailable GNSS is treated as no-fix."""

    def __init__(self, port: str, baud: int = 9600):
        self.port = port
        self.baud = baud
        self.latest: Position | None = None
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import serial
        except ImportError:
            logger.warning("pyserial is not installed; GNSS disabled")
            return
        self.running = True
        self.thread = threading.Thread(target=self._read, args=(serial,), daemon=True)
        self.thread.start()

    def _read(self, serial_module) -> None:
        try:
            with serial_module.Serial(self.port, self.baud, timeout=1) as serial_port:
                while self.running:
                    position = parse_nmea(serial_port.readline().decode("ascii", errors="ignore"))
                    if position:
                        self.latest = position
        except Exception as exc:  # noqa: BLE001
            logger.warning("GNSS reader stopped: %s", exc)

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def snapshot(self) -> Position | None:
        return self.latest