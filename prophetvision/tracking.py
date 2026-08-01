"""Video calibration and tracking of the ball and rotor.

Ball: frame-differencing restricted to the rim annulus; the ball is the
dominant fast-moving bright blob there. Azimuth samples are unwrapped into a
continuous angle signal.

Rotor: azimuthal intensity profile sampled on the pocket ring; consecutive
profiles are phase-correlated (FFT) to measure frame-to-frame rotation with
sub-degree accuracy, then integrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import WheelConfig
from .geometry import sample_ring, circular_shift_deg, unwrap_deg


@dataclass
class Calibration:
    cx: float
    cy: float
    radius: float  # outer wheel radius in pixels

    @classmethod
    def detect(cls, frames: list[np.ndarray]) -> "Calibration":
        """Detect the wheel as the dominant circle in a median frame."""
        med = np.median(np.stack([f.astype(np.float32) for f in frames]), axis=0)
        gray = cv2.cvtColor(med.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        h, w = gray.shape
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w),
            param1=120, param2=40,
            minRadius=int(min(h, w) * 0.2), maxRadius=int(min(h, w) * 0.55),
        )
        if circles is None:
            # Fallback: assume the wheel fills the frame.
            return cls(cx=w / 2.0, cy=h / 2.0, radius=min(h, w) * 0.45)
        x, y, r = circles[0][0]
        return cls(cx=float(x), cy=float(y), radius=float(r))


@dataclass
class TrackResult:
    t: np.ndarray            # sample times (s)
    theta_ball: np.ndarray   # unwrapped ball azimuth (deg), NaN-free
    t_rotor: np.ndarray
    phi_rotor: np.ndarray    # unwrapped rotor azimuth (deg)
    calibration: Calibration = None
    # True when phi_rotor is the absolute azimuth of the reference (zero)
    # pocket, tracked directly off its color marker. False when it is only
    # a relative rotation accumulated by ring correlation.
    phi_is_absolute: bool = False


class SpinTracker:
    """Consumes frames, produces continuous ball/rotor angle tracks."""

    def __init__(self, wheel: WheelConfig, calibration: Calibration,
                 fps: float, n_bins: int = 720):
        self.wheel = wheel
        self.cal = calibration
        self.fps = fps
        self.n_bins = n_bins
        self.prev_gray: np.ndarray | None = None
        self.prev_profile: np.ndarray | None = None
        self.ball_t: list[float] = []
        self.ball_theta_raw: list[float] = []
        self.rotor_t: list[float] = []
        self.rotor_phi_cum: list[float] = []
        self.zero_az_raw: list[float] = []
        self.zero_strength: list[float] = []
        self._phi_acc = 0.0
        self._frame_idx = 0

    # ------------------------------------------------------------------
    def _annulus_mask(self, shape) -> np.ndarray:
        h, w = shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        rr = np.hypot(xx - self.cal.cx, yy - self.cal.cy)
        return ((rr >= self.cal.radius * self.wheel.ball_track_inner) &
                (rr <= self.cal.radius * self.wheel.ball_track_outer))

    def _zero_marker(self, frame: np.ndarray) -> tuple[float, float]:
        """Azimuth (deg) and detection strength of the green zero marker."""
        b, g, r = cv2.split(frame.astype(np.float32))
        greenness = g - 0.5 * (r + b)
        prof = sample_ring(
            greenness, self.cal.cx, self.cal.cy,
            self.cal.radius * self.wheel.rotor_ring_radius,
            self.cal.radius * self.wheel.rotor_ring_width,
            n_bins=self.n_bins,
        )
        n = len(prof)
        win = max(3, int(round(n / self.wheel.n_pockets)))
        kernel = np.zeros(n)
        kernel[:win] = 1.0 / win
        kernel = np.roll(kernel, -(win // 2))
        sm = np.real(np.fft.ifft(np.fft.fft(prof) * np.fft.fft(kernel)))
        k = int(np.argmax(sm))
        strength = float((sm[k] - np.median(sm)) / (np.std(sm) + 1e-9))
        # Sub-bin refinement: circular centroid around the peak.
        offs = np.arange(-win, win + 1)
        idx = (k + offs) % n
        w = np.clip(prof[idx] - np.median(prof), 0.0, None)
        center = k if w.sum() <= 0 else k + float((offs * w).sum() / w.sum())
        return (center * 360.0 / n) % 360.0, strength

    def process(self, frame: np.ndarray) -> None:
        t = self._frame_idx / self.fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # --- rotor: absolute zero-marker azimuth (primary)
        az, strength = self._zero_marker(frame)
        self.zero_az_raw.append(az)
        self.zero_strength.append(strength)

        # --- rotor: ring profile phase correlation
        profile = sample_ring(
            gray, self.cal.cx, self.cal.cy,
            self.cal.radius * self.wheel.rotor_ring_radius,
            self.cal.radius * self.wheel.rotor_ring_width,
            n_bins=self.n_bins,
        )
        if self.prev_profile is not None:
            self._phi_acc += circular_shift_deg(self.prev_profile, profile,
                                                max_shift_deg=0.9 * 360.0
                                                / self.wheel.n_pockets)
        self.rotor_t.append(t)
        self.rotor_phi_cum.append(self._phi_acc)
        self.prev_profile = profile

        # --- ball: frame differencing in the rim annulus
        if self.prev_gray is not None:
            if not hasattr(self, "_mask"):
                self._mask = self._annulus_mask(gray.shape)
            diff = np.abs(gray - self.prev_gray)
            diff[~self._mask] = 0.0
            peak = float(diff.max())
            if peak > 25.0:  # motion energy threshold
                ys, xs = np.nonzero(diff > 0.6 * peak)
                wts = diff[ys, xs]
                bx = float(np.average(xs, weights=wts))
                by = float(np.average(ys, weights=wts))
                theta = float(np.degrees(np.arctan2(by - self.cal.cy,
                                                    bx - self.cal.cx)))
                self.ball_t.append(t)
                self.ball_theta_raw.append(theta)
        self.prev_gray = gray
        self._frame_idx += 1

    # ------------------------------------------------------------------
    def result(self) -> TrackResult:
        t = np.asarray(self.ball_t)
        theta = unwrap_deg(np.asarray(self.ball_theta_raw))
        # Reject isolated mis-detections: the unwrapped track must be close
        # to monotone; drop samples whose local step deviates wildly.
        if len(t) > 8:
            step = np.diff(theta)
            med = np.median(step)
            good = np.ones(len(t), dtype=bool)
            bad_steps = np.abs(step - med) > max(8 * np.abs(med), 45.0)
            good[1:][bad_steps] = False
            t, theta = t[good], unwrap_deg(np.asarray(self.ball_theta_raw)[good])
        # Prefer the drift-free absolute zero-marker track when the marker
        # is reliably visible; otherwise fall back to correlation.
        absolute = (len(self.zero_strength) > 0
                    and float(np.median(self.zero_strength)) > 3.0)
        phi = (unwrap_deg(np.asarray(self.zero_az_raw)) if absolute
               else np.asarray(self.rotor_phi_cum))
        return TrackResult(
            t=t, theta_ball=theta,
            t_rotor=np.asarray(self.rotor_t),
            phi_rotor=phi,
            calibration=self.cal,
            phi_is_absolute=absolute,
        )


def find_reference_pocket_azimuth(frame: np.ndarray, cal: Calibration,
                                  wheel: WheelConfig) -> float:
    """Azimuth (deg, image convention) of the green zero pocket in `frame`.

    Looks for the greenest azimuthal sector on the pocket ring.
    """
    b, g, r = cv2.split(frame.astype(np.float32))
    greenness = g - 0.5 * (r + b)
    prof = sample_ring(greenness, cal.cx, cal.cy,
                       cal.radius * wheel.rotor_ring_radius,
                       cal.radius * wheel.rotor_ring_width, n_bins=720)
    # Smooth over roughly one pocket width.
    n = len(prof)
    win = max(3, int(n / wheel.n_pockets))
    kernel = np.ones(win) / win
    sm = np.real(np.fft.ifft(np.fft.fft(prof) * np.fft.fft(
        np.roll(np.pad(kernel, (0, n - win)), -(win // 2)))))
    return float(np.argmax(sm) * (360.0 / n))
