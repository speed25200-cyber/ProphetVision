"""Side-view (oblique camera) support.

The core predictor assumes a top-down camera, where the wheel is a circle and
image azimuth equals wheel azimuth. From an oblique viewpoint the wheel
projects to an **ellipse**, and image azimuth is a badly distorted function of
the true wheel angle: near the ends of the minor axis a given wheel-angle step
compresses to a few pixels, while along the major axis it stretches. Fitting
the decay model on raw image angles therefore produces a spurious periodic
error at exactly one cycle per revolution, which is the worst possible shape
of error for extrapolating a drop point.

This module removes that distortion. It recovers the wheel plane from the
observed ellipse and maps image points back to true wheel coordinates
(radius fraction, azimuth), so the existing tracking and physics code can run
unchanged on oblique footage.

Two camera models are provided:

* ``affine`` — scaled-orthographic. The circle's tilt is read directly from
  the axis ratio. Exact for a distant camera (long lens), needs nothing but
  the fitted ellipse.
* ``homography`` — full projective. Needed when the camera is close enough
  that the near rim is visibly larger than the far rim. It is calibrated from
  the ellipse plus the *offset between the ellipse centre and the projected
  wheel centre*, which is what perspective actually produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EllipseCalibration:
    """Wheel geometry as seen by an oblique camera.

    ``cx, cy`` is the centre of the fitted ellipse, ``a``/``b`` its semi-major
    and semi-minor axes in pixels, ``phi_deg`` the image-plane rotation of the
    major axis. ``hub_dx, hub_dy`` is the offset (pixels) from the ellipse
    centre to the *observed* wheel hub; under pure orthographic projection it
    is zero, and its magnitude drives the perspective correction.
    """

    cx: float
    cy: float
    a: float
    b: float
    phi_deg: float
    hub_dx: float = 0.0
    hub_dy: float = 0.0
    model: str = "affine"

    # ------------------------------------------------------------------
    @property
    def tilt_deg(self) -> float:
        """Camera elevation below top-down, in degrees (0 = top-down)."""
        ratio = float(np.clip(self.b / max(self.a, 1e-9), 0.0, 1.0))
        return float(np.degrees(np.arccos(ratio)))

    @property
    def radius(self) -> float:
        """Wheel radius in wheel-plane pixels (the un-foreshortened scale)."""
        return float(self.a)

    def _basis(self):
        p = np.radians(self.phi_deg)
        c, s = np.cos(p), np.sin(p)
        # Major-axis unit vector and minor-axis unit vector in image space.
        return np.array([c, s]), np.array([-s, c])

    # ------------------------------------------------------------------
    def to_image(self, r_frac, theta_deg):
        """Wheel coords (radius fraction, azimuth deg) -> image pixel."""
        th = np.radians(np.asarray(theta_deg, dtype=float))
        r = np.asarray(r_frac, dtype=float)
        u = r * np.cos(th) * self.a          # along major axis
        v = r * np.sin(th) * self.b          # along minor axis (compressed)
        e1, e2 = self._basis()
        x = self.cx + u * e1[0] + v * e2[0]
        y = self.cy + u * e1[1] + v * e2[1]
        if self.model == "homography":
            # Perspective: points on the near side (positive minor-axis
            # component) are magnified about the hub, far side shrunk.
            k = self._persp_k()
            depth = 1.0 + k * (r * np.sin(th))
            hx, hy = self.cx + self.hub_dx, self.cy + self.hub_dy
            x = hx + (x - hx) / depth
            y = hy + (y - hy) / depth
        return x, y

    def _persp_k(self) -> float:
        """Perspective strength inferred from the hub/centre offset.

        For a tilted circle under perspective the ellipse centre shifts away
        from the true projected centre along the minor axis by roughly
        k * b / (1 - k^2); inverting to first order gives k from the observed
        offset.
        """
        _, e2 = self._basis()
        off = self.hub_dx * e2[0] + self.hub_dy * e2[1]
        return float(np.clip(-off / max(self.b, 1e-9), -0.6, 0.6))

    def to_wheel(self, x, y):
        """Image pixel -> (radius fraction, azimuth deg) in the wheel plane."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.model == "homography":
            k = self._persp_k()
            hx, hy = self.cx + self.hub_dx, self.cy + self.hub_dy
            dx, dy = x - hx, y - hy
            e1, e2 = self._basis()
            # Undo the depth scaling: solve depth from the projected minor
            # component, since depth depends only on it.
            m = dx * e2[0] + dy * e2[1]          # projected minor component
            # m = (r sin th * b) / depth, depth = 1 + k r sin th
            # let q = r sin th  ->  m = q b / (1 + k q)  ->  q = m / (b - k m)
            q = m / (self.b - k * m + 1e-12)
            depth = 1.0 + k * q
            x = hx + dx * depth
            y = hy + dy * depth
        dx, dy = x - self.cx, y - self.cy
        e1, e2 = self._basis()
        u = dx * e1[0] + dy * e1[1]
        v = dx * e2[0] + dy * e2[1]
        cu = u / self.a
        cv_ = v / self.b
        r = np.hypot(cu, cv_)
        th = np.degrees(np.arctan2(cv_, cu)) % 360.0
        return r, th

    # ------------------------------------------------------------------
    @classmethod
    def from_ellipse(cls, ellipse, hub=None, model="affine") -> "EllipseCalibration":
        """Build from an OpenCV ellipse ``((cx,cy),(MA,ma),angle)``."""
        (cx, cy), (MA, ma), ang = ellipse
        a, b = max(MA, ma) / 2.0, min(MA, ma) / 2.0
        # cv2 reports the angle of the *first* axis; normalise to major.
        phi = ang if MA >= ma else ang + 90.0
        hub_dx = hub_dy = 0.0
        if hub is not None:
            hub_dx, hub_dy = float(hub[0]) - cx, float(hub[1]) - cy
        return cls(cx=float(cx), cy=float(cy), a=float(a), b=float(b),
                   phi_deg=float(phi % 180.0), hub_dx=hub_dx, hub_dy=hub_dy,
                   model=model)

    @classmethod
    def detect(cls, frames, model="affine") -> "EllipseCalibration":
        """Fit the wheel ellipse from the moving rim in a set of frames.

        The rotating pocket ring is the strongest source of temporal variation
        in the scene, so its extent traces the wheel outline.
        """
        stack = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
                          for f in frames])
        activity = stack.std(axis=0)
        thr = np.percentile(activity, 99.0)
        mask = (activity > thr).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = [c.reshape(-1, 2) for c in cnts if len(c) >= 20]
        if not pts:
            raise RuntimeError("no rotating structure found to fit an ellipse")
        allpts = np.vstack(pts).astype(np.float32)
        ell = cv2.fitEllipse(allpts)
        hub = allpts.mean(axis=0)
        return cls.from_ellipse(ell, hub=hub, model=model)


# ----------------------------------------------------------------------
def sample_elliptical_ring(gray: np.ndarray, cal: EllipseCalibration,
                           r_frac: float, width: float,
                           n_bins: int = 720) -> np.ndarray:
    """Azimuthal intensity profile along a ring **in wheel coordinates**.

    Bin i covers true wheel azimuth [i, i+1) * 360/n_bins, so a rotor turning
    at constant speed produces a profile that translates at constant rate —
    which is what the correlation-based rotor tracker assumes.
    """
    h, w = gray.shape[:2]
    thetas = np.arange(n_bins) * (360.0 / n_bins)
    acc = np.zeros(n_bins, dtype=np.float64)
    for rr in np.linspace(r_frac - width / 2.0, r_frac + width / 2.0, 5):
        xs, ys = cal.to_image(rr, thetas)
        xs = np.clip(xs.astype(int), 0, w - 1)
        ys = np.clip(ys.astype(int), 0, h - 1)
        acc += gray[ys, xs]
    return acc / 5.0


def elliptical_annulus_mask(shape, cal: EllipseCalibration,
                            r_inner: float, r_outer: float) -> np.ndarray:
    """Boolean mask of the wheel-plane annulus [r_inner, r_outer]."""
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r, _ = cal.to_wheel(xx, yy)
    return (r >= r_inner) & (r <= r_outer)


def visibility_weight(cal: EllipseCalibration, theta_deg) -> np.ndarray:
    """How well a given wheel azimuth is resolved in this view (0..1).

    Near the minor-axis extremes the projection compresses many degrees of
    wheel angle into few pixels, so measurements there are far noisier. The
    weight is the local |d(image arc)/d(theta)| normalised to its maximum, and
    is used to down-weight those samples when fitting.
    """
    th = np.radians(np.asarray(theta_deg, dtype=float))
    # Speed of the projected point per unit wheel angle, on the unit ring:
    # d/dth (a cos th, b sin th) = (-a sin th, b cos th).
    speed = np.hypot(cal.a * np.sin(th), cal.b * np.cos(th))
    return speed / max(cal.a, 1e-9)
