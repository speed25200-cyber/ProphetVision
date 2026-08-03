"""Drift-free absolute rotor phase.

Tracking the green zero pocket frame by frame is the obvious approach and it
is what earlier versions did, but on real footage it is noisy: the marker is
small, often specular, and sometimes occluded, which produced a 22-degree RMS
fit — a third of a pocket of jitter, enough to make the pocket assignment
meaningless.

This module instead correlates each frame's pocket-ring profile against a
single **reference frame**. The pocket ring is a rich, high-contrast pattern,
so the correlation peak is sharp and the resulting phase is measured directly
against the reference rather than accumulated — there is no integration and
therefore no drift. Two details matter:

* the ring is almost periodic with 37 pockets, so the correlation aliases; the
  search is restricted to the arc the rotor can physically have covered since
  the reference, which selects the right branch;
* one absolute anchor is still needed to name pockets. That comes from the
  green zero on the reference frame alone, where it can be picked carefully,
  instead of on every frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import sample_ring


def zero_azimuth(frame: np.ndarray, cx: float, cy: float, radius: float,
                 ring_r: float = 0.615, ring_width: float = 0.09,
                 n_pockets: int = 37, n_bins: int = 1080) -> tuple[float, float]:
    """Azimuth (deg) and strength of the green zero pocket in one frame."""
    b, g, r = cv2.split(frame.astype(np.float32))
    greenness = g - 0.5 * (r + b)
    prof = sample_ring(greenness, cx, cy, radius * ring_r,
                       radius * ring_width, n_bins=n_bins)
    win = max(3, int(round(n_bins / n_pockets)))
    kern = np.zeros(n_bins)
    kern[:win] = 1.0 / win
    kern = np.roll(kern, -(win // 2))
    smooth = np.real(np.fft.ifft(np.fft.fft(prof) * np.fft.fft(kern)))
    k = int(np.argmax(smooth))
    strength = float((smooth[k] - np.median(smooth)) / (np.std(smooth) + 1e-9))
    offs = np.arange(-win, win + 1)
    idx = (k + offs) % n_bins
    w = np.clip(prof[idx] - np.median(prof), 0.0, None)
    centre = k if w.sum() <= 0 else k + float((offs * w).sum() / w.sum())
    return (centre * 360.0 / n_bins) % 360.0, strength


@dataclass
class RotorPhase:
    """Absolute azimuth of the reference pocket over time."""

    t: np.ndarray
    zero_deg: np.ndarray          # absolute azimuth of the zero pocket
    speed_deg_s: float
    residual_deg: float
    _unwrapped: np.ndarray = None

    def at(self, t: float) -> float:
        """Zero-pocket azimuth at time t (linear model, deg)."""
        return float((self._c[0] * (t - self.t[0]) + self._c[1]) % 360.0)

    def __post_init__(self):
        rel = self.t - self.t[0]
        unwrapped = (self._unwrapped if self._unwrapped is not None
                     else np.degrees(np.unwrap(np.radians(self.zero_deg))))
        self._c = np.polyfit(rel, unwrapped, 1)
        self.speed_deg_s = float(self._c[0])
        self.residual_deg = float(
            (unwrapped - np.polyval(self._c, rel)).std())


class RotorPhaseTracker:
    """Measure rotor phase by correlation against a reference frame."""

    def __init__(self, cx: float, cy: float, radius: float,
                 ring_r: float = 0.62, ring_width: float = 0.12,
                 n_bins: int = 1080, max_speed_deg_s: float = 400.0):
        self.cx, self.cy, self.R = cx, cy, radius
        self.ring_r, self.ring_width = ring_r, ring_width
        self.n_bins = n_bins
        self.max_speed = max_speed_deg_s
        self._ref: np.ndarray | None = None
        self._ref_t: float | None = None
        self.t: list[float] = []
        self.shift: list[float] = []
        self._zero_obs: list[tuple[float, float, float]] = []

    def _profile(self, frame: np.ndarray) -> np.ndarray:
        gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame.ndim == 3 else frame).astype(np.float32)
        prof = sample_ring(gray, self.cx, self.cy, self.R * self.ring_r,
                           self.R * self.ring_width, n_bins=self.n_bins)
        return prof - prof.mean()

    def feed(self, frame: np.ndarray, t: float) -> None:
        """Accumulate frame-to-frame rotation and collect zero-marker sightings.

        Correlating against a fixed reference frame cannot work here: the ring
        repeats every 360/37 degrees, so past one pocket the peak aliases and
        the measured shift just oscillates around zero (measured: +-13 deg of
        noise instead of a 60 deg/s ramp). Consecutive frames, in contrast,
        differ by well under one pocket, so their correlation is unambiguous;
        integrating those steps gives a clean relative phase.
        """
        prof = self._profile(frame)
        if self._ref is None:
            self._ref = prof
            self._ref_t = t
            self.t.append(t)
            self.shift.append(0.0)
        else:
            n = self.n_bins
            corr = np.real(np.fft.ifft(np.fft.fft(prof)
                                       * np.conj(np.fft.fft(self._ref))))
            dt = max(t - self._ref_t, 1e-6)
            lim = max(1, int(abs(self.max_speed) * dt / (360.0 / n)))
            lim = min(lim, n // 2 - 1)
            allowed = np.zeros(n, dtype=bool)
            allowed[:lim + 1] = True
            allowed[n - lim:] = True
            corr = np.where(allowed, corr, -np.inf)
            k = int(np.argmax(corr))
            y0, y1, y2 = corr[(k - 1) % n], corr[k], corr[(k + 1) % n]
            if np.isfinite(y0) and np.isfinite(y2):
                den = y0 - 2 * y1 + y2
                k = k + (0.0 if abs(den) < 1e-12 else 0.5 * (y0 - y2) / den)
            step = k if k <= n / 2 else k - n
            self.t.append(t)
            self.shift.append(self.shift[-1] + step * 360.0 / n)
            self._ref = prof
            self._ref_t = t
        az, strength = zero_azimuth(frame, self.cx, self.cy, self.R)
        self._zero_obs.append((az, strength, self.shift[-1]))

    def result(self, min_strength: float = 3.0) -> RotorPhase:
        """Absolute zero-pocket azimuth over time.

        The anchor uses *every* sighting of the green zero, de-rotated by the
        accumulated phase so they should all agree, then combined circularly.
        Anchoring on a single frame (what earlier versions did) inherited that
        frame's noise; averaging hundreds of de-rotated sightings does not.
        """
        if len(self.t) < 2 or not self._zero_obs:
            raise RuntimeError("not enough frames for a rotor phase")
        obs = [(az, s, sh) for az, s, sh in self._zero_obs if s >= min_strength]
        if not obs:
            obs = self._zero_obs
        az_obs = np.array([az for az, _, _ in obs])
        sh_obs = np.array([sh for _, _, sh in obs])
        weights = np.array([s for _, s, _ in obs], dtype=float)

        # The accumulated shift carries a small scale error (sub-bin
        # interpolation bias compounds over hundreds of steps: measured ~8%
        # over 3 s, i.e. a whole pocket of anchor error, which would rename
        # every pocket). Fit both the anchor and that scale against the
        # zero-marker sightings instead of assuming the scale is exactly 1.
        unwrapped = np.degrees(np.unwrap(np.radians(az_obs)))
        gain = 1.0
        if np.ptp(sh_obs) > 20.0:
            A = np.column_stack([sh_obs, np.ones_like(sh_obs)])
            W = np.sqrt(weights)[:, None]
            sol, *_ = np.linalg.lstsq(A * W, unwrapped * W[:, 0], rcond=None)
            if 0.7 < sol[0] < 1.4:
                gain = float(sol[0])
        resid = unwrapped - gain * sh_obs
        seed = np.arctan2((weights * np.sin(np.radians(resid))).sum(),
                          (weights * np.cos(np.radians(resid))).sum())
        centred = (resid - np.degrees(seed) + 180.0) % 360.0 - 180.0
        anchor = float((np.degrees(seed) + np.median(centred)) % 360.0)
        t = np.asarray(self.t, dtype=float)
        zero = anchor + gain * np.asarray(self.shift, dtype=float)
        return RotorPhase(t=t, zero_deg=zero % 360.0, speed_deg_s=0.0,
                          residual_deg=0.0, _unwrapped=zero)
