"""Tests for prophetvision/rotortrack.py (SPEC section E).

Streaming rotor tracking on the plunge sequence of hq_060_100
(clip t = 21.5-39.5 s), calibration cx=610, cy=419, R=345.
Measured ground truth: rotor speed ~= 58 deg/s, green zero marker visible
(absolute tracking).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from prophetvision.rotortrack import RotorTracker
from prophetvision.geometry import smooth_derivative

VIDEOS = "/mnt/agents/output/prophetvision/videos"
CLIP_A = os.path.join(VIDEOS, "hq_060_100.mp4")

pytestmark = pytest.mark.skipif(not os.path.exists(CLIP_A),
                                reason="real video window not available")

FPS = 60.0
CX, CY, R = 610.0, 419.0, 345.0


@dataclass
class _CalR:  # v2 style: R attribute
    cx: float = CX
    cy: float = CY
    R: float = R


@dataclass
class _CalRadius:  # v1 style: radius attribute
    cx: float = CX
    cy: float = CY
    radius: float = R


def run_tracker(segments, cal=None):
    """Feed the listed (t0, t1) clip segments frame by frame (streaming)."""
    tracker = RotorTracker(fps=FPS)
    tracker.set_calibration(cal or _CalR())
    cap = cv2.VideoCapture(CLIP_A)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / FPS
        if any(t0 <= t <= t1 for t0, t1 in segments):
            tracker.process(t, frame)
        i += 1
        if t > max(t1 for _, t1 in segments):
            break
    cap.release()
    return tracker


# ----------------------------------------------------------------------
def test_absolute_track_and_speed():
    tracker = run_tracker([(21.5, 39.5)])
    t, phi, is_absolute = tracker.track()
    assert is_absolute, "zero marker should give an absolute track"
    assert len(t) > 500
    assert np.all(np.isfinite(phi))
    # median instantaneous rotor speed ~= 58 deg/s (+/- 15)
    omega = smooth_derivative(t, phi, window=15)
    median_speed = float(np.median(omega))
    assert abs(median_speed - 58.0) <= 15.0, \
        f"median rotor speed {median_speed:.1f} deg/s"
    # total arc is consistent (several full turns over 18 s)
    assert phi[-1] - phi[0] > 500.0


def test_calibration_accepts_radius_attribute():
    tracker = RotorTracker(fps=FPS)
    tracker.set_calibration(_CalRadius())  # must not raise
    tracker.set_calibration({"cx": CX, "cy": CY, "radius": R})
    tracker.set_calibration((CX, CY, R))


def test_process_requires_calibration():
    tracker = RotorTracker(fps=FPS)
    frame = np.zeros((910, 1206, 3), np.uint8)
    with pytest.raises(RuntimeError):
        tracker.process(0.0, frame)


def test_soft_reinit_after_camera_cut():
    """A cut to an oblique shot and back must not corrupt the track:
    no azimuth jump larger than physically possible, still absolute."""
    tracker = run_tracker([(21.5, 28.0), (10.0, 10.3), (28.0, 34.0)])
    t, phi, is_absolute = tracker.track()
    assert np.all(np.isfinite(phi))
    steps = np.abs(np.diff(phi))
    assert steps.max() <= 25.0, f"azimuth jump after cut: {steps.max():.1f}"
    omega = smooth_derivative(t, phi, window=15)
    median_speed = float(np.median(omega))
    assert abs(median_speed - 58.0) <= 20.0, \
        f"median rotor speed after cut {median_speed:.1f} deg/s"
