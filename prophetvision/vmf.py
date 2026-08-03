"""Velocity-matched filtering (track-before-detect) for faint fast movers.

Per-frame blob detection fails on this footage: in the oblique/side view the
ball is small, dim, heavily motion-blurred and often occluded, so in any single
frame it is below the noise. Every detector built on "find the blob, then link
the blobs" — frame differencing, brightness thresholding, colour signature,
polar unwrap plus median background — was defeated by it (see README).

The standard answer to that problem is **track-before-detect**: never threshold
a single frame. Instead, hypothesise a trajectory, integrate the raw signal
*along* it, and threshold the integral. A real mover adds coherently over N
frames (signal ~ N, noise ~ sqrt(N), so SNR grows as sqrt(N)); anything else
averages out. Here the trajectory family is

    theta(t) = phi + omega * t + 0.5 * alpha * t^2

on the azimuth axis of a polar (or elliptical-polar) unwrap, so the integral is
a shift-and-add over (phi, omega, alpha) — a Radon transform of the space-time
map. That is what :func:`velocity_matched_filter` computes.

Two auxiliary pieces make it usable on a real wheel:

* :func:`null_rotor` — the rotor's own texture (pockets, numbers, turret arms)
  is far brighter than the ball and would dominate the transform. Shifting each
  row into the rotor's rotating frame makes that texture time-invariant, so a
  temporal median removes it exactly; shifting back leaves everything that is
  *not* rotor-locked, the ball included.

* :func:`detector_power` — the blindness control. A negative result only means
  something if the detector can find a ball it is *known* to contain. This runs
  the same filter over a window with a confirmed ball and reports the achieved
  peak-to-floor ratio, so "no detection" can be reported as either
  "no mover present" or "detector below threshold here" — never confused.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ----------------------------------------------------------------------
@dataclass
class SpaceTime:
    """Azimuth-vs-time map of a ring, in wheel-plane azimuth."""

    S: np.ndarray        # (n_frames, n_bins) intensity
    t: np.ndarray        # (n_frames,) seconds
    n_bins: int

    @property
    def deg_per_bin(self) -> float:
        return 360.0 / self.n_bins

    def detrended(self) -> "SpaceTime":
        """Remove static azimuthal structure and per-frame brightness drift."""
        S = self.S - np.median(self.S, axis=0)
        S = S - S.mean(axis=1, keepdims=True)
        return SpaceTime(S=S, t=self.t, n_bins=self.n_bins)


def build_spacetime(frames_iter, to_image, r_lo: float, r_hi: float,
                    n_bins: int = 720, n_rad: int = 8,
                    shape: tuple[int, int] | None = None) -> SpaceTime:
    """Sample a ring into an azimuth profile per frame.

    ``to_image(r_frac, theta_deg) -> (x, y)`` maps wheel coordinates to pixels;
    pass a circular mapping for top-down or an
    :class:`~prophetvision.sideview.EllipseCalibration` for oblique views, so
    the azimuth axis is always *wheel* azimuth.
    """
    thetas = np.arange(n_bins) * (360.0 / n_bins)
    radii = np.linspace(r_lo, r_hi, n_rad)
    px = py = None
    rows, times = [], []
    for t, frame in frames_iter:
        gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame.ndim == 3 else frame).astype(np.float32)
        if px is None:
            h, w = gray.shape[:2] if shape is None else shape
            xs, ys = [], []
            for rr in radii:
                x, y = to_image(rr, thetas)
                xs.append(np.clip(np.asarray(x).astype(int), 0, w - 1))
                ys.append(np.clip(np.asarray(y).astype(int), 0, h - 1))
            px, py = np.array(xs), np.array(ys)
        rows.append(gray[py, px].max(axis=0))
        times.append(float(t))
    if not rows:
        raise ValueError("no frames supplied")
    return SpaceTime(S=np.asarray(rows), t=np.asarray(times), n_bins=n_bins)


# ----------------------------------------------------------------------
def measure_rotor(st: SpaceTime, max_speed_deg_s: float = 400.0) -> float:
    """Dominant rigid rotation rate of the map, deg/s.

    Static structure is removed first (otherwise the correlation peak sits at
    zero shift and the measurement silently returns 0), and the peak is
    interpolated to sub-bin accuracy.

    A pocket ring is almost periodic, so the frame-to-frame correlation aliases:
    a true rate w is indistinguishable from w + k * n_pockets * fps * sector.
    ``max_speed_deg_s`` restricts the search to the physically plausible band
    and picks the correct branch.
    """
    D = st.S - np.median(st.S, axis=0)
    n = st.n_bins
    speeds = []
    for k in range(1, len(D)):
        a = D[k - 1] - D[k - 1].mean()
        b = D[k] - D[k].mean()
        corr = np.real(np.fft.ifft(np.fft.fft(b) * np.conj(np.fft.fft(a))))
        dt = st.t[k] - st.t[k - 1]
        if dt > 0 and np.isfinite(max_speed_deg_s):
            lim = max(1, int(abs(max_speed_deg_s) * dt / st.deg_per_bin))
            allowed = np.zeros(n, dtype=bool)
            allowed[:lim + 1] = True
            allowed[n - lim:] = True
            corr = np.where(allowed, corr, -np.inf)
        j = int(np.argmax(corr))
        y0, y1, y2 = corr[(j - 1) % n], corr[j], corr[(j + 1) % n]
        if not (np.isfinite(y0) and np.isfinite(y2)):
            frac = 0.0
        else:
            den = y0 - 2 * y1 + y2
            frac = 0.0 if abs(den) < 1e-12 else 0.5 * (y0 - y2) / den
        shift = j + frac
        if shift > n / 2:
            shift -= n
        if dt > 0:
            speeds.append(shift * st.deg_per_bin / dt)
    return float(np.median(speeds)) if speeds else 0.0


def null_rotor(st: SpaceTime, omega_rotor: float | None = None) -> SpaceTime:
    """Remove everything rotating rigidly at ``omega_rotor`` deg/s."""
    if omega_rotor is None:
        omega_rotor = measure_rotor(st)
    rel = st.t - st.t[0]
    shift = np.round(omega_rotor * rel / st.deg_per_bin).astype(int)
    rot = np.array([np.roll(st.S[k], -shift[k]) for k in range(len(st.S))])
    rot = rot - np.median(rot, axis=0)
    back = np.array([np.roll(rot[k], shift[k]) for k in range(len(rot))])
    back = back - back.mean(axis=1, keepdims=True)
    return SpaceTime(S=back, t=st.t, n_bins=st.n_bins)


# ----------------------------------------------------------------------
@dataclass
class VMFResult:
    snr: float             # peak / std of the accumulated profile
    floor: float           # median peak-SNR over the whole hypothesis grid
    omega: float           # deg/s
    alpha: float           # deg/s^2
    phi: float             # deg, azimuth at t[0]
    ranking: list          # [(snr, omega, alpha, phi), ...] best first

    @property
    def ratio(self) -> float:
        """Peak SNR relative to the grid's own noise floor."""
        return self.snr / max(self.floor, 1e-9)

    def is_detection(self, min_ratio: float = 1.8,
                     min_isolation: float = 1.15) -> bool:
        """A detection must beat the floor *and* stand apart from unrelated
        hypotheses. Scattered near-equal peaks at wildly different velocities
        are the signature of noise, not of a mover."""
        if self.ratio < min_ratio:
            return False
        rivals = [r for r in self.ranking[1:]
                  if abs(r[1] - self.omega) > 150.0]
        if not rivals:
            return True
        return self.snr >= min_isolation * rivals[0][0]


def velocity_matched_filter(st: SpaceTime, omegas: np.ndarray,
                            alphas: np.ndarray = np.array([0.0]),
                            ) -> VMFResult:
    """Integrate along every (omega, alpha) trajectory; return the best."""
    rel = st.t - st.t[0]
    out = []
    for a in np.atleast_1d(alphas):
        for w in np.atleast_1d(omegas):
            shift = np.round((w * rel + 0.5 * a * rel * rel)
                             / st.deg_per_bin).astype(int)
            acc = np.zeros(st.n_bins)
            for k in range(len(rel)):
                acc += np.roll(st.S[k], -shift[k])
            acc -= acc.mean()
            sd = acc.std()
            snr = float(acc.max() / sd) if sd > 0 else 0.0
            out.append((snr, float(w), float(a),
                        float(np.argmax(acc) * st.deg_per_bin)))
    out.sort(reverse=True)
    floor = float(np.median([r[0] for r in out]))
    best = out[0]
    return VMFResult(snr=best[0], floor=floor, omega=best[1], alpha=best[2],
                     phi=best[3], ranking=out)


def detector_power(st_known: SpaceTime, omegas: np.ndarray,
                   alphas: np.ndarray = np.array([0.0]),
                   expected_omega: float | None = None,
                   tol: float = 150.0) -> tuple[bool, VMFResult]:
    """Blindness control: can the filter find a ball known to be present?

    Returns ``(can_see, result)``. When ``can_see`` is False, a null result on
    any comparable window is uninformative — the detector, not the video, is
    the limiting factor. Reporting a negative without running this is how three
    earlier detectors in this project produced confident wrong answers.
    """
    res = velocity_matched_filter(st_known, omegas, alphas)
    ok = res.is_detection()
    if ok and expected_omega is not None:
        ok = abs(res.omega - expected_omega) <= tol
    return ok, res
