"""Plunge (circle) and oblique (ellipse) calibration. SPEC.md section B."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PlungeCal:
    """Calibration of the top-down (plunge) shot: wheel rim circle."""

    cx: float
    cy: float
    R: float

    @classmethod
    def detect(cls, frame: np.ndarray) -> "PlungeCal | None":
        """Detect the rim circle via Hough (R in [250, 420] px, centered on
        the image). Returns None when no centered circle is found."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=w // 2,
            param1=100,
            param2=45,
            minRadius=250,
            maxRadius=420,
        )
        if circles is None:
            return None
        best = None
        for c in circles[0]:
            cx, cy, r = float(c[0]), float(c[1]), float(c[2])
            dc = float(np.hypot(cx - w / 2.0, cy - h / 2.0))
            if best is None or dc < best[0]:
                best = (dc, cx, cy, r)
        # The plunge rim circle is essentially centered on the image; reject
        # off-center spurious circles (e.g. oblique shots).
        if best is None or best[0] > 0.12 * min(w, h):
            return None
        return cls(cx=best[1], cy=best[2], R=best[3])


@dataclass
class ObliqueCal:
    """Ellipse of the wheel rim in the oblique shot.

    ``a`` and ``b`` are the FULL major/minor axis lengths (as returned by
    ``cv2.fitEllipse``), ``angle`` is the rotation of the major axis in
    degrees, clockwise in image coordinates (x right, y down) — the same
    convention as ``cv2.ellipse``.
    """

    cx: float
    cy: float
    a: float
    b: float
    angle: float

    # ------------------------------------------------------------- detection
    @classmethod
    def detect(cls, frame: np.ndarray, n_iter: int = 4000) -> "ObliqueCal | None":
        """Fit the rim ellipse: Canny edges in the wheel band, then RANSAC
        over ``cv2.fitEllipse`` on point subsets with plausibility gating
        (wide + flat ellipse, centered), then iterative inlier refinement.

        Returns None when no plausible ellipse is found.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        band = np.zeros_like(edges)
        band[int(0.30 * h):int(0.72 * h), int(0.20 * w):int(0.80 * w)] = 255
        ys, xs = np.nonzero(cv2.bitwise_and(edges, band))
        if len(xs) < 200:
            return None
        pts = np.column_stack([xs, ys]).astype(np.float64)

        tol = 2.5
        best = None
        # A few deterministic RANSAC runs; keep the one with most inliers.
        for seed in range(3):
            cand = cls._ransac(pts, w, h, n_iter=n_iter, seed=seed, tol=tol)
            if cand is not None and (best is None or cand[0] > best[0]):
                best = cand
        if best is None:
            return None
        el, score = cls._refine(pts, best[1], tol=tol)
        if score < 100:
            return None
        return cls._from_cv2(el)

    @staticmethod
    def _plausible(el, w: int, h: int) -> bool:
        (cx, cy), (d1, d2), _ang = el
        a, b = max(d1, d2), min(d1, d2)
        if b <= 0 or a <= 0:
            return False
        # Wide, flat, roughly centered ellipse (rim seen obliquely).
        return (
            0.35 * w <= a <= 0.75 * w
            and 0.20 <= b / a <= 0.45
            and 0.33 * w <= cx <= 0.67 * w
            and 0.42 * h <= cy <= 0.62 * h
        )

    @staticmethod
    def _dist(pts: np.ndarray, el) -> np.ndarray:
        """Approximate distance (px) of points to the ellipse perimeter."""
        (cx, cy), (d1, d2), ang = el
        sa, sb = max(d1, d2) / 2.0, min(d1, d2) / 2.0
        th = np.deg2rad(ang if d1 >= d2 else ang + 90.0)
        ca, sa_ = np.cos(th), np.sin(th)
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        xr = ca * dx + sa_ * dy
        yr = -sa_ * dx + ca * dy
        val = np.sqrt((xr / sa) ** 2 + (yr / sb) ** 2)
        return np.abs(val - 1.0) * (sa + sb) / 2.0

    @classmethod
    def _ransac(cls, pts, w, h, n_iter, seed, tol):
        rng = np.random.default_rng(seed)
        best = None
        n = len(pts)
        for _ in range(n_iter):
            idx = rng.choice(n, size=12, replace=False)
            try:
                el = cv2.fitEllipse(pts[idx].astype(np.float32).reshape(-1, 1, 2))
            except cv2.error:
                continue
            if not cls._plausible(el, w, h):
                continue
            score = int((cls._dist(pts, el) < tol).sum())
            if best is None or score > best[0]:
                best = (score, el)
        return best

    @classmethod
    def _refine(cls, pts, el, tol, rounds=4):
        for _ in range(rounds):
            inl = pts[cls._dist(pts, el) < tol]
            if len(inl) < 20:
                break
            el = cv2.fitEllipse(inl.astype(np.float32).reshape(-1, 1, 2))
        return el, int((cls._dist(pts, el) < tol).sum())

    @staticmethod
    def _from_cv2(el) -> "ObliqueCal":
        (cx, cy), (d1, d2), ang = el
        if d1 >= d2:
            a, b, angle = d1, d2, ang
        else:
            a, b, angle = d2, d1, ang + 90.0
        angle = float(angle % 180.0)
        return ObliqueCal(cx=float(cx), cy=float(cy), a=float(a), b=float(b), angle=angle)

    # --------------------------------------------------------------- mapping
    def to_image(self, azimuth_deg: float, r_frac: float) -> tuple[float, float]:
        """Canonical polar point (azimuth in degrees, radius fraction of the
        rim) -> image (x, y).

        The canonical wheel circle (azimuth measured with atan2 in image
        convention, y down) is scaled by the semi-axes (a/2, b/2) and rotated
        by ``angle`` — the inverse of the ellipse->circle normalization.
        """
        th = np.deg2rad(azimuth_deg)
        ux, uy = np.cos(th) * r_frac, np.sin(th) * r_frac  # canonical circle
        ph = np.deg2rad(self.angle)
        ca, sa = np.cos(ph), np.sin(ph)
        ex, ey = ux * self.a / 2.0, uy * self.b / 2.0  # axis-aligned ellipse
        x = self.cx + ca * ex - sa * ey
        y = self.cy + sa * ex + ca * ey
        return (float(x), float(y))

    def to_azimuth(self, x: float, y: float) -> float:
        """Image point -> canonical azimuth in degrees [0, 360), consistent
        with atan2 in image coordinates (y down): inverse of ``to_image``.
        """
        ph = np.deg2rad(self.angle)
        ca, sa = np.cos(ph), np.sin(ph)
        dx, dy = x - self.cx, y - self.cy
        ex = ca * dx + sa * dy
        ey = -sa * dx + ca * dy
        ux, uy = ex / (self.a / 2.0), ey / (self.b / 2.0)
        return float(np.rad2deg(np.arctan2(uy, ux)) % 360.0)
