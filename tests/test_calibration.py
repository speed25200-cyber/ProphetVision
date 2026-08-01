"""Tests for prophetvision.calibration (SPEC B) on the real test videos."""
from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from prophetvision.calibration import ObliqueCal, PlungeCal

VIDEOS = "/mnt/agents/output/prophetvision/videos"
CLIP_A = os.path.join(VIDEOS, "hq_060_100.mp4")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(CLIP_A), reason="test videos not available"
)


def grab(path: str, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"cannot read frame at t={t}"
    return frame


# ----------------------------------------------------------------- PlungeCal
def test_plunge_detect_on_plunge_frames():
    """Measured ground truth: center=(610, 419), R=345 px."""
    for t in (22.5, 25.0, 30.0, 35.0, 38.5):
        cal = PlungeCal.detect(grab(CLIP_A, t))
        assert cal is not None, f"no plunge circle at t={t}"
        assert cal.cx == pytest.approx(610.0, abs=10.0)
        assert cal.cy == pytest.approx(419.0, abs=10.0)
        assert cal.R == pytest.approx(345.0, abs=10.0)


def test_plunge_detect_none_on_oblique():
    for t in (5.0, 10.0, 15.0):
        assert PlungeCal.detect(grab(CLIP_A, t)) is None


def test_plunge_detect_none_on_blank():
    assert PlungeCal.detect(np.zeros((910, 1206, 3), np.uint8)) is None


# ---------------------------------------------------------------- ObliqueCal
def test_oblique_detect_plausible_ellipse():
    """Measured: center ~(605, 480), major axis ~640 px, minor axis ~230 px."""
    cals = []
    for t in (5.0, 10.0, 15.0):
        cal = ObliqueCal.detect(grab(CLIP_A, t))
        assert cal is not None, f"no oblique ellipse at t={t}"
        cals.append(cal)
    a = float(np.median([c.a for c in cals]))
    b = float(np.median([c.b for c in cals]))
    cx = float(np.median([c.cx for c in cals]))
    cy = float(np.median([c.cy for c in cals]))
    assert a > 500.0
    assert b < 300.0
    assert b < a
    assert abs(cx - 605.0) < 80.0
    assert abs(cy - 480.0) < 80.0


def test_oblique_detect_none_on_blank():
    assert ObliqueCal.detect(np.zeros((910, 1206, 3), np.uint8)) is None


def test_oblique_mapping_roundtrip():
    cal = ObliqueCal(cx=605.0, cy=480.0, a=640.0, b=230.0, angle=7.0)
    for az in (0.0, 30.0, 90.0, 135.0, 179.9, 200.0, 271.0, 359.0):
        for r in (0.2, 0.55, 1.0):
            x, y = cal.to_image(az, r)
            err = (cal.to_azimuth(x, y) - az + 180.0) % 360.0 - 180.0
            assert err == pytest.approx(0.0, abs=1e-6)


def test_oblique_to_image_matches_cv2_ellipse():
    """Points from to_image(r=1) must lie on the cv2-drawn ellipse: the
    angle convention is the cv2 one (clockwise, y down)."""
    cal = ObliqueCal(cx=605.0, cy=480.0, a=640.0, b=230.0, angle=13.0)
    mask = np.zeros((910, 1206), np.uint8)
    cv2.ellipse(mask, ((cal.cx, cal.cy), (cal.a, cal.b), cal.angle), 255, 2)
    pts = np.array([cal.to_image(az, 1.0) for az in range(0, 360, 2)])
    xi = np.round(pts[:, 0]).astype(int)
    yi = np.round(pts[:, 1]).astype(int)
    on = mask[yi, xi] > 0
    assert on.mean() > 0.9


def test_oblique_detected_cal_mapping_roundtrip():
    cal = ObliqueCal.detect(grab(CLIP_A, 10.0))
    assert cal is not None
    x, y = cal.to_image(123.0, 0.9)
    assert 0 <= x < 1206 and 0 <= y < 910
    assert cal.to_azimuth(x, y) == pytest.approx(123.0, abs=1e-6)
