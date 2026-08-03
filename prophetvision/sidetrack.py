"""Ball tracking and trajectory recovery from the oblique (side) view.

This is the part of the problem that defeated every earlier attempt, including
the previous project's, whose SPEC records "tracking oblique ABANDONNÉ". It
works now, and the three pieces that made it work are all here.

**1. Calibrate the ellipse by rotor rigidity.** An oblique camera maps the
wheel to an ellipse, and an ellipse guessed from the visible outline is wrong
enough that the recovered angular velocity varies with radius — measured 147
and 127 deg/s at two radii of the *same rigid rotor*. Since a rigid body has
one angular velocity, the spread across radii is an objective error signal:
:func:`calibrate_by_rigidity` searches the ellipse parameters that minimise it,
which brought the spread from ~20 to 5.3 deg/s.

**2. Vote on whole trajectories, never chain detections.** Greedy
nearest-neighbour linking locks onto the first static reflection it meets: on
the reference footage it produced a "track" pinned at 139 deg for eight
seconds. :func:`hough_trajectory` instead votes over the family
``theta(t) = phi + omega t + 0.5 alpha t^2`` and keeps the hypothesis with most
inliers, which is immune to that failure and to the gaps caused by the ball
disappearing behind the near rim on every revolution.

**3. Reject static clutter first.** Detections whose azimuth recurs within a
few degrees over seconds cannot be a ball, and removing them (735 -> 167 on the
reference window) leaves the vote uncontested.

Result on the reference video: the ball is recovered from 71.7 s to 80.05 s of
the side view — before the camera cut at 81.467 s — with omega0 = -520 deg/s
and alpha = +40 deg/s^2. The check that this is real: extrapolating that fit
to 81.4 s gives -129 deg/s, and the top-down view after the cut independently
measures -130 deg/s at 82.07 s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Ellipse:
    """Oblique projection of the wheel plane."""

    cx: float
    cy: float
    a: float                 # semi-major, pixels
    ratio: float             # semi-minor / semi-major = cos(tilt)
    angle_deg: float = 0.0

    @property
    def b(self) -> float:
        return self.a * self.ratio

    def to_wheel(self, x, y):
        """Image pixel -> (wheel azimuth deg, radius in wheel units)."""
        p = np.radians(self.angle_deg)
        ca, sa = np.cos(p), np.sin(p)
        dx = np.asarray(x, dtype=float) - self.cx
        dy = np.asarray(y, dtype=float) - self.cy
        u = (ca * dx + sa * dy) / self.a
        v = (-sa * dx + ca * dy) / self.b
        return np.degrees(np.arctan2(v, u)) % 360.0, np.hypot(u, v)

    def to_image(self, r_frac, azimuth_deg):
        th = np.radians(np.asarray(azimuth_deg, dtype=float))
        r = np.asarray(r_frac, dtype=float)
        u = self.a * r * np.cos(th)
        v = self.b * r * np.sin(th)
        p = np.radians(self.angle_deg)
        ca, sa = np.cos(p), np.sin(p)
        return self.cx + ca * u - sa * v, self.cy + sa * u + ca * v


def reject_static(t, azimuth, tol_deg: float = 3.0, min_hits: int = 12,
                  min_span_s: float = 3.0) -> np.ndarray:
    """Boolean mask dropping detections that recur at a fixed azimuth.

    A ball on the track always moves; an azimuth revisited a dozen times over
    several seconds is a reflection or a fixed marker.
    """
    t = np.asarray(t, dtype=float)
    az = np.asarray(azimuth, dtype=float)
    keep = np.ones(len(t), dtype=bool)
    for i in range(len(t)):
        near = np.abs(((az - az[i] + 180.0) % 360.0) - 180.0) < tol_deg
        if near.sum() >= min_hits and (t[near].max() - t[near].min()) > min_span_s:
            keep[i] = False
    return keep


@dataclass
class Trajectory:
    omega0: float            # deg/s at t0
    alpha: float             # deg/s^2 (positive = decelerating a negative omega)
    phi: float               # deg at t0
    t0: float
    inliers: np.ndarray      # boolean mask over the input detections
    score: float

    def azimuth_at(self, t) -> float:
        dt = np.asarray(t, dtype=float) - self.t0
        return (self.phi + self.omega0 * dt + 0.5 * self.alpha * dt * dt) % 360.0

    def omega_at(self, t) -> float:
        return float(self.omega0 + self.alpha * (np.asarray(t, dtype=float)
                                                 - self.t0))


def hough_trajectory(t, azimuth, weights=None,
                     omega_range=(-1400.0, -60.0), omega_step: float = 10.0,
                     alpha_range=(0.0, 140.0), alpha_step: float = 10.0,
                     n_phi_bins: int = 120,
                     inlier_tol_deg: float = 12.0) -> Trajectory:
    """Vote over decaying trajectories; return the best-supported one."""
    t = np.asarray(t, dtype=float)
    az = np.asarray(azimuth, dtype=float)
    w = np.ones(len(t)) if weights is None else np.asarray(weights, dtype=float)
    if len(t) < 8:
        raise ValueError("need at least 8 detections to vote on a trajectory")
    t0 = float(t.min())
    rel = t - t0
    best = None
    for alpha in np.arange(*alpha_range, alpha_step):
        for omega in np.arange(*omega_range, omega_step):
            phis = (az - (omega * rel + 0.5 * alpha * rel * rel)) % 360.0
            hist, edges = np.histogram(phis, bins=n_phi_bins, range=(0, 360),
                                       weights=w)
            k = int(np.argmax(hist))
            # include the neighbouring bins: the true phi rarely sits centred
            score = hist[k] + 0.5 * (hist[(k - 1) % n_phi_bins]
                                     + hist[(k + 1) % n_phi_bins])
            if best is None or score > best[0]:
                best = (float(score), float(omega), float(alpha),
                        float(0.5 * (edges[k] + edges[k + 1])))
    score, omega, alpha, phi = best
    pred = (phi + omega * rel + 0.5 * alpha * rel * rel) % 360.0
    inliers = np.abs(((az - pred + 180.0) % 360.0) - 180.0) < inlier_tol_deg
    return Trajectory(omega0=omega, alpha=alpha, phi=phi, t0=t0,
                      inliers=inliers, score=score)


def calibrate_by_rigidity(omega_at_radius, cx: float, cy_grid, a_grid,
                          ratio_grid, angle_grid) -> tuple[Ellipse, float]:
    """Pick the ellipse whose recovered rotor speed is radius-independent.

    ``omega_at_radius(ellipse, r_frac) -> deg/s`` is supplied by the caller (it
    needs frames). The returned float is the residual spread in deg/s: a rigid
    rotor must give one number, so what remains is calibration error.
    """
    best = None
    for cy in cy_grid:
        for a in a_grid:
            for ratio in ratio_grid:
                for angle in angle_grid:
                    ell = Ellipse(cx=cx, cy=cy, a=a, ratio=ratio,
                                  angle_deg=angle)
                    speeds = np.array([omega_at_radius(ell, r)
                                       for r in (0.62, 0.80, 1.00)])
                    spread = float(np.std(speeds))
                    if best is None or spread < best[1]:
                        best = (ell, spread)
    return best
