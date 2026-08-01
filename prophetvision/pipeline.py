"""High-level entry points: frames/video in, prediction out."""

from __future__ import annotations

import cv2
import numpy as np

from .config import WheelConfig
from .predict import LandingZonePredictor, Prediction
from .tracking import (Calibration, SpinTracker, TrackResult,
                       find_reference_pocket_azimuth)


def read_video(path: str, max_seconds: float | None = None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    limit = int(max_seconds * fps) if max_seconds else None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if limit and len(frames) >= limit:
            break
    cap.release()
    if not frames:
        raise IOError(f"no frames decoded from: {path}")
    return frames, fps


def analyze_frames(frames, fps: float,
                   predictor: LandingZonePredictor | None = None,
                   wheel: WheelConfig | None = None,
                   cutoff_seconds: float | None = None,
                   calibration: Calibration | None = None,
                   ) -> tuple[Prediction, TrackResult, LandingZonePredictor]:
    """Run the full pipeline on decoded frames.

    `cutoff_seconds` limits how much of the spin the predictor is allowed to
    see (to demonstrate genuine ahead-of-time prediction). The prediction is
    made from those frames only.
    """
    wheel = wheel or (predictor.wheel if predictor else WheelConfig())
    predictor = predictor or LandingZonePredictor(wheel=wheel)
    cal = calibration or Calibration.detect(frames[: min(15, len(frames))])
    predictor.rotor_zero_azimuth = find_reference_pocket_azimuth(
        frames[0], cal, wheel)

    n_use = len(frames)
    if cutoff_seconds is not None:
        n_use = min(n_use, int(cutoff_seconds * fps))
    tracker = SpinTracker(wheel, cal, fps)
    for f in frames[:n_use]:
        tracker.process(f)
    track = tracker.result()
    if len(track.t) < 12:
        raise RuntimeError(
            "ball track too short: could not detect a moving ball on the rim "
            f"({len(track.t)} samples). Check calibration/annulus settings.")
    pred = predictor.predict(track)
    return pred, track, predictor
