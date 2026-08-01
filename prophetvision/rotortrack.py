"""Streaming rotor tracker (green zero + FFT fallback). SPEC.md section E.

Streaming port of the v1 rotor logic (``prophetvision/tracking.py``):
the absolute azimuth of the green zero pocket is measured on every frame
(primary, drift-free), and a ring-profile FFT phase correlation is
accumulated as fallback when the marker is not reliably visible.

Differences with v1: frames are consumed one at a time via
``process(t, frame)`` (no full-video buffering), calibration can be
(re)set at any time (per-shot recalibration), and camera cuts are
handled by a soft reinit: the correlation accumulator discards the
profile pair straddling the cut and isolated zero-marker outliers are
rejected, so the azimuth track shows no spurious jump.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import WheelConfig
from .geometry import sample_ring, circular_shift_deg, unwrap_deg


class RotorTracker:
    """Streaming rotor azimuth tracker.

    ``set_calibration`` accepts any object with ``cx``, ``cy`` and either
    ``R`` or ``radius`` attributes (e.g. v1 ``Calibration`` or v2
    ``PlungeCal``), or a plain mapping/tuple.
    """

    #: Minimum median zero-marker strength to trust the absolute track.
    ABSOLUTE_STRENGTH = 3.0
    #: Mean |diff| on a 300x226 gray pair above which a camera cut is assumed.
    CUT_DIFF_THRESH = 8.0

    def __init__(self, fps: float, n_bins: int = 720,
                 wheel: WheelConfig | None = None):
        self.fps = float(fps)
        self.n_bins = int(n_bins)
        self.wheel = wheel or WheelConfig()
        self.cal = None
        # per-frame records
        self._t: list[float] = []
        self._zero_az: list[float] = []
        self._zero_strength: list[float] = []
        self._zero_ok: list[bool] = []
        self._phi_acc = 0.0
        self._phi_cum: list[float] = []
        self._prev_profile: np.ndarray | None = None
        self._prev_small: np.ndarray | None = None
        self._last_good_az: float | None = None
        self._median_step: float = 0.0

    # ------------------------------------------------------------------
    def set_calibration(self, cal) -> None:
        """(Re)set wheel calibration; soft-reinits the correlation state."""
        if isinstance(cal, dict):
            cx, cy = cal["cx"], cal["cy"]
            R = cal.get("R", cal.get("radius"))
        elif isinstance(cal, (tuple, list)):
            cx, cy, R = cal
        else:
            cx, cy = cal.cx, cal.cy
            R = getattr(cal, "R", None)
            if R is None:
                R = cal.radius
        self.cal = (float(cx), float(cy), float(R))
        self._prev_profile = None  # do not correlate across recalibration

    # ------------------------------------------------------------------
    def _ring_crop(self, frame: np.ndarray):
        """Square crop tightly containing the rotor ring (cheaper math)."""
        cx, cy, R = self.cal
        rad = R * (self.wheel.rotor_ring_radius
                   + self.wheel.rotor_ring_width) + 4.0
        h, w = frame.shape[:2]
        x0 = max(0, int(cx - rad))
        y0 = max(0, int(cy - rad))
        x1 = min(w, int(cx + rad + 1))
        y1 = min(h, int(cy + rad + 1))
        return frame[y0:y1, x0:x1], (cx - x0, cy - y0)

    def _zero_marker(self, crop: np.ndarray, cc) -> tuple[float, float]:
        """Azimuth (deg) and strength of the green zero marker (v1 logic)."""
        _, _, R = self.cal
        b, g, r = cv2.split(crop.astype(np.float32))
        greenness = g - 0.5 * (r + b)
        prof = sample_ring(greenness, cc[0], cc[1],
                           R * self.wheel.rotor_ring_radius,
                           R * self.wheel.rotor_ring_width,
                           n_bins=self.n_bins)
        n = len(prof)
        win = max(3, int(round(n / self.wheel.n_pockets)))
        kernel = np.zeros(n)
        kernel[:win] = 1.0 / win
        kernel = np.roll(kernel, -(win // 2))
        sm = np.real(np.fft.ifft(np.fft.fft(prof) * np.fft.fft(kernel)))
        k = int(np.argmax(sm))
        strength = float((sm[k] - np.median(sm)) / (np.std(sm) + 1e-9))
        offs = np.arange(-win, win + 1)
        idx = (k + offs) % n
        w = np.clip(prof[idx] - np.median(prof), 0.0, None)
        center = k if w.sum() <= 0 else k + float((offs * w).sum() / w.sum())
        return (center * 360.0 / n) % 360.0, strength

    # ------------------------------------------------------------------
    def process(self, t: float, frame: np.ndarray) -> None:
        if self.cal is None:
            raise RuntimeError("set_calibration() must be called first")
        cx, cy, R = self.cal
        small = cv2.cvtColor(cv2.resize(frame, (300, 226)),
                             cv2.COLOR_BGR2GRAY)

        # Camera-cut detection (frank cut => large normalized frame diff).
        cut = False
        if self._prev_small is not None:
            diff = float(np.abs(small.astype(np.float32)
                                - self._prev_small.astype(np.float32)).mean())
            cut = diff > self.CUT_DIFF_THRESH
        if cut:
            self._prev_profile = None  # soft reinit, no spurious shift

        # --- absolute zero-marker azimuth (primary)
        crop, cc = self._ring_crop(frame)
        az, strength = self._zero_marker(crop, cc)
        ok = strength > 1.5
        if ok and self._last_good_az is not None and not cut:
            step = (az - self._last_good_az + 180.0) % 360.0 - 180.0
            # Reject isolated outliers: the rotor cannot jump more than
            # ~15 deg between accepted samples.
            max_step = max(15.0, 4.0 * abs(self._median_step))
            if abs(step) > max_step:
                ok = False
        if ok:
            self._last_good_az = az
        self._t.append(float(t))
        self._zero_az.append(az)
        self._zero_strength.append(strength)
        self._zero_ok.append(ok)

        # --- ring-profile FFT phase correlation (fallback, v1 logic)
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        profile = sample_ring(gray_crop.astype(np.float32), cc[0], cc[1],
                              R * self.wheel.rotor_ring_radius,
                              R * self.wheel.rotor_ring_width,
                              n_bins=self.n_bins)
        if self._prev_profile is not None:
            self._phi_acc += circular_shift_deg(
                self._prev_profile, profile,
                max_shift_deg=0.9 * 360.0 / self.wheel.n_pockets)
        self._phi_cum.append(self._phi_acc)
        self._prev_profile = profile
        self._prev_small = small

    # ------------------------------------------------------------------
    def track(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return (t, phi_unwrapped_deg, is_absolute).

        Absolute mode (zero marker reliably visible): unwrapped azimuth of
        the green zero pocket, outliers removed. Fallback: accumulated
        ring-correlation rotation (relative, drift possible).
        """
        t = np.asarray(self._t, dtype=float)
        strengths = np.asarray(self._zero_strength, dtype=float)
        absolute = (len(strengths) > 0
                    and float(np.median(strengths)) > self.ABSOLUTE_STRENGTH)
        if absolute:
            ok = np.asarray(self._zero_ok, dtype=bool)
            az = np.asarray(self._zero_az, dtype=float)
            if ok.sum() >= 2:
                phi = unwrap_deg(az[ok])
                t_out = t[ok]
                # Soft-reinit repair: a step larger than what the rotor can
                # physically do between two accepted samples means a camera
                # cut moved the reference; re-anchor the track so it stays
                # continuous (no azimuth jump).
                steps = np.diff(phi)
                if len(steps):
                    self._median_step = float(np.median(steps))
                    max_step = max(20.0, 6.0 * abs(self._median_step))
                    excess = steps - np.clip(steps, -max_step, max_step)
                    phi = phi - np.concatenate(([0.0], np.cumsum(excess)))
                return t_out, phi, True
            return t, unwrap_deg(az), True
        return t, np.asarray(self._phi_cum, dtype=float), False
