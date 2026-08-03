"""Real-footage spin analysis: what actually worked on live wheel video.

This module encodes the pipeline validated on real footage (see README):

* **Ball tracking** — the ball is found as a *yellow-cream, moving* blob:
  yellowness = min(G, R) - B, minus a temporal median background (which
  removes every static glint and reflection, the dominant failure mode of
  naive brightness or frame-difference trackers), restricted to a radial band.
* **Duplicate-frame removal** — broadcast/screen-capture chains duplicate
  frames; duplicates carry no information and corrupt derivative estimates.
* **Direction-aware unwrap** — the ball's azimuth only advances in one
  direction; steps are unwrapped into (-330, +30] degrees so short detection
  gaps cannot silently flip the direction of travel.
* **Rotor tracking** — the green zero pocket tracked as an absolute azimuth
  (drift-free), fitted linearly.
* **Cutoff prediction** — fit the analytic decay model on samples up to a
  cutoff time, extrapolate the trajectory, and convert azimuths to pockets
  under the rotor at contact time.

Measured on the reference video (full-resolution, spin A): fit residual
0.97 deg over 291 deg of arc; landing azimuth predicted 1.2 s ahead with a
7.2 deg error = 0.74 pocket; predicted impact pocket == observed impact
pocket. See README for the full accounting, including a weaker spin where a
detection gap degraded the fit — the feasibility thresholds exist precisely
to flag that case.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import WheelConfig
from .physics import BallDecayModel
from .geometry import sample_ring


# ----------------------------------------------------------------------
def unwrap_directional(az_deg: np.ndarray, direction: int = -1,
                       slack_deg: float = 30.0) -> np.ndarray:
    """Unwrap azimuth samples knowing the sign of travel.

    Each step is mapped into (-360+slack, +slack] for direction=-1 (or
    [-slack, 360-slack) for +1), so gaps shorter than one revolution cannot
    flip the apparent direction the way nearest-wrap unwrapping can.
    """
    az = np.asarray(az_deg, dtype=float)
    out = [az[0]]
    for a in az[1:]:
        step = (a - out[-1]) % 360.0
        if direction < 0:
            step = step - 360.0 if step > slack_deg else step
        else:
            step = step if step < 360.0 - slack_deg else step - 360.0
        out.append(out[-1] + step)
    return np.array(out)


def unwrap_predictive(t, az_deg, direction: int = -1,
                      seed_span: float = 0.4) -> np.ndarray:
    """Unwrap azimuths across gaps using the running angular velocity.

    A fixed-slack unwrap (:func:`unwrap_directional`) assumes every gap is
    shorter than one revolution. That is false here: a detector dropout of a
    few tenths of a second while the ball turns at several hundred deg/s hides
    one or more whole turns, and the track silently loses them — the failure
    that made fitted arcs jump from 84 to 253 degrees when a few samples were
    added.

    This version keeps a velocity estimate from the recent accepted samples and
    across each step picks the number of wraps closest to the predicted travel
    ``omega * dt``, so whole revolutions inside a gap are recovered.
    """
    t = np.asarray(t, dtype=float)
    az = np.asarray(az_deg, dtype=float)
    if len(t) < 2:
        return az.copy()
    out = [az[0]]
    omega = None
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            out.append(out[-1])
            continue
        if omega is None:
            # Seed with the direction-constrained step until enough span.
            step = (az[i] - out[-1]) % 360.0
            if direction < 0:
                step -= 360.0 if step > 30.0 else 0.0
            elif step > 330.0:
                step -= 360.0
        else:
            predicted = out[-1] + omega * dt
            # nearest congruent value to the prediction
            step = (az[i] - predicted) % 360.0
            if step > 180.0:
                step -= 360.0
            step += predicted - out[-1]
        out.append(out[-1] + step)
        span = t[i] - t[0]
        if span >= seed_span:
            j = np.searchsorted(t, t[i] - seed_span)
            j = max(0, min(j, i - 1))
            if t[i] > t[j]:
                omega = (out[i] - out[j]) / (t[i] - t[j])
    return np.array(out)


@dataclass
class BallSample:
    t: float
    azimuth_deg: float
    r_frac: float
    strength: float


class YellowBallTracker:
    """Two-pass tracker: collect yellowness maps, subtract the temporal
    median, then pick the strongest moving blob per frame in a radial band."""

    def __init__(self, cx: float, cy: float, radius: float,
                 r_inner: float = 0.78, r_outer: float = 1.12,
                 diff_threshold: float = 16.0, min_score: float = 300.0):
        self.cx, self.cy, self.R = cx, cy, radius
        self.r_inner, self.r_outer = r_inner, r_outer
        self.diff_threshold = diff_threshold
        self.min_score = min_score
        self._maps: list[np.ndarray] = []
        self._times: list[float] = []
        self._prev_gray: np.ndarray | None = None
        self._band: np.ndarray | None = None

    def feed(self, frame: np.ndarray, t: float) -> None:
        b, g, r = cv2.split(frame.astype(np.float32))
        if self._prev_gray is not None and \
                float(np.abs(g - self._prev_gray).mean()) <= 0.01:
            self._prev_gray = g
            return  # duplicated frame
        self._prev_gray = g
        if self._band is None:
            h, w = g.shape
            yy, xx = np.mgrid[0:h, 0:w]
            rf = np.hypot(xx - self.cx, yy - self.cy) / self.R
            self._band = ((rf > self.r_inner) & (rf < self.r_outer)
                          ).astype(np.float32)
        self._maps.append((np.minimum(g, r) - b) * self._band)
        self._times.append(t)

    def samples(self) -> list[BallSample]:
        if len(self._maps) < 5:
            return []
        Y = np.array(self._maps)
        bg = np.median(Y, axis=0)
        out = []
        for k, t in enumerate(self._times):
            D = Y[k] - bg
            m = (D > self.diff_threshold).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            nl, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
            best = None
            for j in range(1, nl):
                area = int(st[j, cv2.CC_STAT_AREA])
                if not (4 <= area <= 5000):
                    continue
                score = float(D[lab == j].sum())
                if best is None or score > best[0]:
                    best = (score, float(ce[j][0]), float(ce[j][1]))
            if best and best[0] >= self.min_score:
                score, px, py = best
                out.append(BallSample(
                    t=t,
                    azimuth_deg=float(np.degrees(
                        np.arctan2(py - self.cy, px - self.cx)) % 360.0),
                    r_frac=float(np.hypot(px - self.cx, py - self.cy) / self.R),
                    strength=score))
        return out


class ZeroMarkerTracker:
    """Absolute rotor azimuth from the green zero pocket, per frame."""

    def __init__(self, cx: float, cy: float, radius: float,
                 ring_r: float = 0.615, ring_width: float = 0.09,
                 n_pockets: int = 37, n_bins: int = 1080):
        self.cx, self.cy, self.R = cx, cy, radius
        self.ring_r, self.ring_width = ring_r, ring_width
        self.n_bins = n_bins
        win = max(3, int(round(n_bins / n_pockets)))
        kern = np.zeros(n_bins)
        kern[:win] = 1.0 / win
        self._fk = np.fft.fft(np.roll(kern, -(win // 2)))
        self._win = win
        self.t: list[float] = []
        self.az: list[float] = []
        self.strength: list[float] = []

    def feed(self, frame: np.ndarray, t: float) -> None:
        b, g, r = cv2.split(frame.astype(np.float32))
        greenness = g - 0.5 * (r + b)
        prof = sample_ring(greenness, self.cx, self.cy,
                           self.R * self.ring_r, self.R * self.ring_width,
                           n_bins=self.n_bins)
        sm = np.real(np.fft.ifft(np.fft.fft(prof) * self._fk))
        k = int(np.argmax(sm))
        offs = np.arange(-self._win, self._win + 1)
        idx = (k + offs) % self.n_bins
        wgt = np.clip(prof[idx] - np.median(prof), 0.0, None)
        c = k if wgt.sum() <= 0 else k + float((offs * wgt).sum() / wgt.sum())
        self.t.append(t)
        self.az.append((c * 360.0 / self.n_bins) % 360.0)
        self.strength.append(float(sm[k] - np.median(sm)))

    def fit(self):
        """Linear rotor model: returns (speed deg/s, azimuth(t) callable, rms)."""
        t = np.asarray(self.t)
        az = np.asarray(self.az)
        s = np.asarray(self.strength)
        good = s > np.median(s) * 0.5
        # nearest-wrap unwrap is fine here: the rotor is slow and steady.
        u = [az[0]]
        for a in az[1:]:
            u.append(u[-1] + ((a - u[-1] + 180.0) % 360.0 - 180.0))
        u = np.asarray(u)
        coef = np.polyfit(t[good] - t[0], u[good], 1)
        rms = float((u[good] - np.polyval(coef, t[good] - t[0])).std())
        t0 = t[0]
        return float(coef[0]), (lambda tt: float(np.polyval(coef, tt - t0) % 360.0)), rms


# ----------------------------------------------------------------------
@dataclass
class SpinPrediction:
    model: BallDecayModel
    rotor_speed: float
    rotor_rms: float
    fit_samples: int
    fit_arc_deg: float
    fit_residual_deg: float
    cutoff: float
    zero_at: object  # callable t -> zero azimuth deg

    def azimuth_at(self, t: float) -> float:
        return float(self.model.theta_at(t) % 360.0)

    def pocket_under(self, azimuth_deg: float, t: float,
                     wheel: WheelConfig) -> tuple[int, float]:
        sect = 360.0 / wheel.n_pockets
        rel = (azimuth_deg - self.zero_at(t)) % 360.0
        return wheel.pocket_order[int(round(rel / sect)) % wheel.n_pockets], rel / sect


def predict_spin(ball: list[BallSample], rotor: ZeroMarkerTracker,
                 cutoff: float, direction: int = -1,
                 rim_r_min: float = 0.85) -> SpinPrediction:
    """Fit the decay model on rim samples up to `cutoff` and package the
    prediction. Raises if the fit window is too thin to mean anything."""
    rim = [s for s in ball if s.t <= cutoff and s.r_frac >= rim_r_min]
    if len(rim) < 20:
        raise RuntimeError(f"only {len(rim)} rim samples before cutoff")
    t = np.array([s.t for s in rim])
    th = unwrap_directional(np.array([s.azimuth_deg for s in rim]), direction)
    model = BallDecayModel.fit(t, th)
    speed, zero_at, rms = rotor.fit()
    return SpinPrediction(
        model=model, rotor_speed=speed, rotor_rms=rms,
        fit_samples=len(rim), fit_arc_deg=float(abs(th[-1] - th[0])),
        fit_residual_deg=model.residual_rms, cutoff=cutoff, zero_at=zero_at)
