"""End-to-end landing-zone prediction: video in, pocket probabilities out.

Chains the pieces that were each validated separately:

    YOLO detections (balldetect, radial-gated)
      -> directional unwrap into a continuous azimuth track
      -> analytic decay fit (physics.BallDecayModel) on samples up to a cutoff
      -> extrapolation to rotor contact
      -> rotor phase (realstream.ZeroMarkerTracker) -> impact pocket
      -> learned scatter (scatter.ScatterModel) -> P(final pocket)

The radial gate is applied *before* non-maximum suppression, which is the
detail that makes the detector usable: the gold turret is round, bright and
ball-coloured, so on a top-down window with the ball certainly on the rim an
ungated sweep returned 417 turret detections at 0.4-0.6 R and 3 on the rim.
Gating first turns that into one clean detection per frame at ~0.83 confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .balldetect import BallDetector, radial_gate
from .config import WheelConfig
from .physics import BallDecayModel
from .predict import LandingZonePredictor, Prediction
from .realstream import ZeroMarkerTracker, unwrap_predictive
from .tracking import TrackResult


@dataclass
class SpinReport:
    prediction: Prediction
    n_detections: int
    fit_samples: int
    fit_arc_deg: float
    fit_residual_deg: float
    cutoff: float
    lead_s: float
    rotor_speed: float
    mean_confidence: float

    def summary(self, zone_width: int = 9) -> dict:
        out = self.prediction.summary(zone_width=zone_width)
        out.update({
            "detections": self.n_detections,
            "fit_samples": self.fit_samples,
            "fit_arc_deg": round(self.fit_arc_deg, 1),
            "fit_residual_deg": round(self.fit_residual_deg, 2),
            "cutoff_s": round(self.cutoff, 2),
            "lead_s": round(self.lead_s, 2),
            "rotor_deg_s": round(self.rotor_speed, 1),
            "mean_confidence": round(self.mean_confidence, 2),
        })
        return out


def iter_unique_frames(path: str, t0: float = 0.0, t1: float | None = None,
                       time_offset: float = 0.0):
    """Yield (t, frame) for frames that actually carry new content.

    Broadcast and screen-capture chains duplicate frames; duplicates add no
    information and corrupt every derivative estimate downstream.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
    stop = None if t1 is None else int(t1 * fps)
    prev = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if stop is not None and idx >= stop:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None and float(np.abs(gray - prev).mean()) <= 0.01:
                prev = gray
                continue
            prev = gray
            yield time_offset + idx / fps, frame
    finally:
        cap.release()


def predict_spin(path: str, cx: float, cy: float, radius: float,
                 cutoff: float, t_contact: float,
                 t0: float = 0.0, t1: float | None = None,
                 time_offset: float = 0.0,
                 model_path: str = "ball_sota_yolo11n_320_fp16.onnx",
                 wheel: WheelConfig | None = None,
                 predictor: LandingZonePredictor | None = None,
                 r_gate: tuple[float, float] = (0.80, 1.10),
                 conf_threshold: float = 0.35,
                 min_arc_deg: float = 250.0) -> SpinReport:
    """Predict the landing zone from a top-down window.

    Only detections at or before ``cutoff`` feed the fit; ``t_contact`` is when
    the ball is expected to reach the rotor, so ``t_contact - cutoff`` is the
    genuine prediction lead.
    """
    wheel = wheel or WheelConfig()
    predictor = predictor or LandingZonePredictor(wheel=wheel)
    detector = BallDetector(model_path=model_path,
                            conf_threshold=conf_threshold)
    gate = radial_gate(cx, cy, radius, radius, *r_gate)
    rotor = ZeroMarkerTracker(cx, cy, radius)

    times, azimuths, confs = [], [], []
    for t, frame in iter_unique_frames(path, t0, t1, time_offset):
        rotor.feed(frame, t)
        if t > cutoff:
            continue
        dets = detector.detect_frame(
            frame, t=t,
            roi=(max(0, int(cx - 1.4 * radius)), max(0, int(cy - 1.4 * radius)),
                 int(cx + 1.4 * radius), int(cy + 1.4 * radius)),
            accept=gate)
        if not dets:
            continue
        best = max(dets, key=lambda d: d.confidence)
        times.append(t)
        azimuths.append(float(np.degrees(
            np.arctan2(best.y - cy, best.x - cx)) % 360.0))
        confs.append(best.confidence)

    if len(times) < 20:
        raise RuntimeError(f"only {len(times)} gated detections before cutoff")

    t_arr = np.asarray(times)
    az = np.asarray(azimuths)
    # Direction of travel from the median signed step, then a unwrap that
    # cannot flip across a detection gap.
    steps = (np.diff(az) + 180.0) % 360.0 - 180.0
    direction = 1 if np.median(steps) >= 0 else -1
    theta = unwrap_predictive(t_arr, az, direction=direction)

    arc = float(abs(theta[-1] - theta[0]))
    if arc < min_arc_deg:
        # Measured on the reference video: below ~250 deg of tracked arc the
        # two decay coefficients are not separable and the extrapolation error
        # jumps from ~0.2 to 3-5 pockets. Refuse rather than return a number
        # the data cannot support.
        raise RuntimeError(
            f"tracked arc {arc:.0f} deg is below the {min_arc_deg:.0f} deg "
            "needed to identify the decay model")

    ball = BallDecayModel.fit(t_arr, theta)
    speed, zero_at, _ = rotor.fit()

    track = TrackResult(t=t_arr, theta_ball=theta,
                        t_rotor=np.asarray(rotor.t),
                        phi_rotor=np.asarray([zero_at(tt) for tt in rotor.t]),
                        phi_is_absolute=True)
    predictor.fall_time = max(t_contact - cutoff, 0.0)
    predictor.omega_drop = max(abs(ball.omega_at(t_contact)), 1.0)
    predictor.fall_travel_deg = 0.0
    pred = predictor.predict(track)

    return SpinReport(
        prediction=pred, n_detections=len(times), fit_samples=len(t_arr),
        fit_arc_deg=float(abs(theta[-1] - theta[0])),
        fit_residual_deg=float(ball.residual_rms), cutoff=cutoff,
        lead_s=float(t_contact - cutoff), rotor_speed=float(speed),
        mean_confidence=float(np.mean(confs)))
