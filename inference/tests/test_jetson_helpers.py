import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from camera import build_csi_pipeline
from camera import open_video
from gnss import parse_nmea


def test_csi_pipeline_uses_latest_frame_settings():
    pipeline = build_csi_pipeline(sensor_id=1, width=1920, height=1080, fps=30)
    assert "sensor-id=1" in pipeline
    assert "width=1920,height=1080,framerate=30/1" in pipeline
    assert "drop=true max-buffers=1" in pipeline


def test_open_video_uses_file_path():
    class FakeCapture:
        def isOpened(self):
            return True

    class FakeCv2:
        CAPTURED_PATH = None

        @staticmethod
        def VideoCapture(path):
            FakeCv2.CAPTURED_PATH = path
            return FakeCapture()

    capture = open_video(FakeCv2, "../demo.mp4")
    assert FakeCv2.CAPTURED_PATH == "../demo.mp4"
    assert isinstance(capture, FakeCapture)


def test_parse_gga_position():
    position = parse_nmea("$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47")
    assert position is not None
    assert round(position.latitude, 5) == 48.1173
    assert round(position.longitude, 5) == 11.51667


def test_parse_rmc_southern_western_position():
    position = parse_nmea("$GPRMC,225446,A,4916.45,S,12311.12,W,000.5,054.7,191194,020.3,E*68")
    assert position is not None
    assert position.latitude < 0
    assert position.longitude < 0


def test_parse_no_fix_returns_none():
    assert parse_nmea("$GPGGA,123519,4807.038,N,01131.000,E,0,00,0.9,545.4,M,46.9,M,,*47") is None
    assert parse_nmea("not-nmea") is None