"""Angle utilities and polar sampling helpers."""

from __future__ import annotations

import numpy as np


def wrap_deg(a):
    """Wrap angle(s) to [0, 360)."""
    return np.mod(a, 360.0)


def wrap_signed_deg(a):
    """Wrap angle(s) to [-180, 180)."""
    return np.mod(np.asarray(a) + 180.0, 360.0) - 180.0


def unwrap_deg(angles: np.ndarray) -> np.ndarray:
    """Unwrap a sequence of angles in degrees into a continuous signal."""
    return np.degrees(np.unwrap(np.radians(np.asarray(angles, dtype=float))))


def sample_ring(gray: np.ndarray, cx: float, cy: float, radius: float,
                width: float, n_bins: int = 720) -> np.ndarray:
    """Average image intensity over a thin annulus, binned by azimuth.

    Returns a 1-D profile of length `n_bins`, bin i covering azimuth
    [i, i+1) * 360/n_bins degrees (image convention: x right, y down,
    angle measured from +x axis toward +y).
    """
    h, w = gray.shape[:2]
    radii = np.linspace(radius - width / 2.0, radius + width / 2.0, 5)
    thetas = np.radians(np.arange(n_bins) * (360.0 / n_bins))
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    acc = np.zeros(n_bins, dtype=np.float64)
    cnt = np.zeros(n_bins, dtype=np.float64)
    for r in radii:
        xs = np.clip((cx + r * cos_t).astype(int), 0, w - 1)
        ys = np.clip((cy + r * sin_t).astype(int), 0, h - 1)
        acc += gray[ys, xs]
        cnt += 1.0
    return acc / cnt


def circular_shift_deg(profile_a: np.ndarray, profile_b: np.ndarray,
                       max_shift_deg: float | None = None) -> float:
    """Rotation (degrees) that best maps profile_a onto profile_b.

    Uses FFT cross-correlation with parabolic sub-bin peak interpolation.
    Positive result means b is a rotated (toward increasing azimuth) a.
    `max_shift_deg` restricts the search window, which suppresses aliasing
    on azimuthally periodic patterns (e.g. alternating pocket colors).
    """
    a = profile_a - profile_a.mean()
    b = profile_b - profile_b.mean()
    n = len(a)
    corr = np.real(np.fft.ifft(np.fft.fft(b) * np.conj(np.fft.fft(a))))
    if max_shift_deg is not None:
        max_bins = max(2, int(max_shift_deg / (360.0 / n)))
        allowed = np.zeros(n, dtype=bool)
        allowed[:max_bins + 1] = True
        allowed[-max_bins:] = True
        corr = np.where(allowed, corr, -np.inf)
    k = int(np.argmax(corr))
    # Parabolic interpolation around the peak for sub-bin accuracy.
    y0, y1, y2 = corr[(k - 1) % n], corr[k], corr[(k + 1) % n]
    if not (np.isfinite(y0) and np.isfinite(y2)):
        delta = 0.0
    else:
        denom = (y0 - 2 * y1 + y2)
        delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (y0 - y2) / denom
    shift_bins = k + delta
    if shift_bins > n / 2:
        shift_bins -= n
    return shift_bins * (360.0 / n)


def smooth_derivative(t: np.ndarray, y: np.ndarray, window: int = 9,
                      order: int = 2) -> np.ndarray:
    """Derivative dy/dt via local polynomial fits (Savitzky-Golay style,
    tolerant of slightly non-uniform sampling)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    half = window // 2
    dy = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        tt, yy = t[lo:hi], y[lo:hi]
        if len(tt) < order + 1:
            dy[i] = np.gradient(y, t)[i]
            continue
        c = np.polyfit(tt - tt.mean(), yy, order)
        dy[i] = np.polyval(np.polyder(c), t[i] - tt.mean())
    return dy
