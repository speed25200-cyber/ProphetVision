"""Feasibility audit: can a given video support landing-zone prediction at all?

Prediction is only meaningful if the ball can be observed **while the bet is
still open** (or, for a physical wheel, while there is still time to act) and
for long enough that the decay model is identifiable. This module measures
those preconditions and reports them honestly, instead of returning a
confident-looking number that the data cannot support.

Three preconditions are checked:

1. **Observation window** — is the ball visible on the rim, and for how long?
2. **Sampling** — how many *unique* frames cover that window? Video containers
   routinely duplicate frames (a 60 fps file carrying a 25 fps stream), which
   destroys effective temporal resolution.
3. **Identifiability** — over the observed arc, can (c0, c2) be separated well
   enough that extrapolation to the drop point is stable? This is answered by
   refitting under measurement noise and reporting the spread of the predicted
   drop time, converted to pockets.

A run that fails any of these is reported as NOT PREDICTABLE, with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import WheelConfig
from .physics import BallDecayModel
from .tracking import Calibration


@dataclass
class FeasibilityReport:
    ball_visible_s: float = 0.0
    ball_arc_deg: float = 0.0
    unique_samples: int = 0
    effective_fps: float = 0.0
    container_fps: float = 0.0
    drop_time_spread_s: float = float("inf")
    pocket_spread: float = float("inf")
    reasons: list[str] = field(default_factory=list)

    @property
    def predictable(self) -> bool:
        return not self.reasons

    def summary(self) -> dict:
        return {
            "predictable": self.predictable,
            "ball_visible_s": round(self.ball_visible_s, 3),
            "ball_arc_deg": round(self.ball_arc_deg, 1),
            "unique_samples": self.unique_samples,
            "container_fps": round(self.container_fps, 1),
            "effective_unique_fps": round(self.effective_fps, 1),
            "drop_time_spread_s": (None if not np.isfinite(self.drop_time_spread_s)
                                   else round(self.drop_time_spread_s, 3)),
            "landing_spread_pockets": (None if not np.isfinite(self.pocket_spread)
                                       else round(self.pocket_spread, 1)),
            "blocking_reasons": self.reasons,
        }


# Minimum data needed before a two-parameter decay fit extrapolates usefully.
MIN_VISIBLE_S = 1.5
MIN_UNIQUE_SAMPLES = 30
MIN_ARC_DEG = 720.0
MAX_POCKET_SPREAD = 6.0


def audit(video_path: str, wheel: WheelConfig | None = None,
          start_s: float = 0.0, end_s: float | None = None,
          calibration: Calibration | None = None,
          noise_deg: float = 0.5, trials: int = 12) -> FeasibilityReport:
    """Measure whether `video_path` can support prediction."""
    wheel = wheel or WheelConfig()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rep = FeasibilityReport(container_fps=fps)

    i0 = int(start_s * fps)
    i1 = int(end_s * fps) if end_s else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)

    probe = []
    for _ in range(min(20, i1 - i0)):
        ok, f = cap.read()
        if ok:
            probe.append(f)
    if not probe:
        raise IOError("no frames decoded")
    cal = calibration or Calibration.detect(probe)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)

    # Sample the rim annulus into an azimuth profile per frame.
    n_az = 720
    a = np.radians(np.arange(n_az) * (360.0 / n_az))
    radii = np.linspace(wheel.ball_track_inner, wheel.ball_track_outer, 6)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    px = np.clip((cal.cx + cal.radius * np.outer(radii, np.cos(a))).astype(int), 0, w - 1)
    py = np.clip((cal.cy + cal.radius * np.outer(radii, np.sin(a))).astype(int), 0, h - 1)

    strips, times = [], []
    for i in range(i0, i1):
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        strips.append(g[py, px].max(axis=0))
        times.append(i / fps)
    cap.release()
    if len(strips) < 4:
        rep.reasons.append("too few frames decoded to assess")
        return rep

    S = np.asarray(strips)
    t = np.asarray(times)

    # Duplicated frames carry no new information.
    changed = np.r_[True, np.abs(np.diff(S, axis=0)).mean(axis=1) > 0.01]
    span = max(t[-1] - t[0], 1e-6)
    rep.effective_fps = float(changed.sum() / span)

    # The ball is what moves against a static background.
    D = S - np.median(S, axis=0)
    idx = np.nonzero(changed)[0]
    det_t, det_az = [], []
    for k in idx:
        j = int(np.argmax(D[k]))
        if D[k, j] > 25:
            det_t.append(t[k])
            det_az.append(j * (360.0 / n_az))

    if len(det_t) < 4:
        rep.reasons.append("no ball detected on the rim in this window")
        return rep

    det_t = np.asarray(det_t)
    det_az = np.asarray(det_az)
    keep = np.r_[True, np.diff(det_az) != 0]
    det_t, det_az = det_t[keep], det_az[keep]
    theta = np.degrees(np.unwrap(np.radians(det_az)))

    rep.unique_samples = int(len(det_t))
    rep.ball_visible_s = float(det_t[-1] - det_t[0])
    rep.ball_arc_deg = float(abs(theta[-1] - theta[0]))

    if rep.ball_visible_s < MIN_VISIBLE_S:
        rep.reasons.append(
            f"ball observable for only {rep.ball_visible_s:.2f}s "
            f"(need >= {MIN_VISIBLE_S}s to identify the decay model)")
    if rep.unique_samples < MIN_UNIQUE_SAMPLES:
        rep.reasons.append(
            f"only {rep.unique_samples} unique ball samples "
            f"(need >= {MIN_UNIQUE_SAMPLES}); container reports "
            f"{fps:.0f} fps but only {rep.effective_fps:.0f} fps carry new data")
    if rep.ball_arc_deg < MIN_ARC_DEG:
        rep.reasons.append(
            f"ball observed over {rep.ball_arc_deg:.0f} deg of arc "
            f"(need >= {MIN_ARC_DEG:.0f} deg, i.e. a few revolutions)")

    # Identifiability under measurement noise.
    if rep.unique_samples >= 6:
        rng = np.random.default_rng(0)
        drops = []
        for _ in range(trials):
            try:
                m = BallDecayModel.fit(
                    det_t, theta + rng.normal(0, noise_deg, len(theta)))
                td = m.time_to_speed(640.0)
                if np.isfinite(td) and 0 <= td - det_t[0] < 60:
                    drops.append(td)
            except Exception:
                pass
        distinct = len(np.unique(np.round(drops, 3))) if drops else 0
        if len(drops) < max(3, trials // 3) or distinct < 2:
            # Either the fit diverges, or every noisy refit collapses onto the
            # same clamped value — both mean the parameters are unidentifiable,
            # not that the prediction is precise.
            rep.drop_time_spread_s = float("inf")
            rep.reasons.append(
                "decay model is unidentifiable on this track: the observed arc "
                "is too short to separate friction from drag")
        else:
            rep.drop_time_spread_s = float(np.ptp(drops))
            # Relative ball/rotor motion converts drop-time error into pockets.
            rel_deg_per_s = 500.0
            rep.pocket_spread = float(
                rep.drop_time_spread_s * rel_deg_per_s / (360.0 / wheel.n_pockets))
            if rep.pocket_spread > MAX_POCKET_SPREAD:
                rep.reasons.append(
                    f"landing uncertainty +-{rep.pocket_spread:.0f} pockets from "
                    f"{noise_deg} deg of tracking noise alone (wheel has "
                    f"{wheel.n_pockets}) - no better than chance")
    return rep
