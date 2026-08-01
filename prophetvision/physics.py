"""Deterministic dynamics models fitted from tracked angular data.

Ball on the rim: the tangential deceleration is well approximated by a
constant (rolling/sliding friction) plus a term quadratic in speed
(aerodynamic drag):

    d(omega)/dt = -(c0 + c2 * omega^2),   omega > 0

This ODE has a closed-form solution, so once (c0, c2) are fitted from a
fraction of a second of tracking, the entire remaining trajectory — and in
particular the time and azimuth at which the ball slows to its drop speed —
is determined analytically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import smooth_derivative


@dataclass
class BallDecayModel:
    c0: float          # constant deceleration, deg/s^2
    c2: float          # quadratic drag coefficient, 1/deg (per (deg/s)^2 * deg/s^2)
    t_ref: float       # reference time (s) of the fit
    omega_ref: float   # |angular speed| at t_ref, deg/s
    theta_ref: float   # unwrapped azimuth at t_ref, deg (signed direction kept)
    direction: float   # +1 or -1: sign of d(theta)/dt
    residual_rms: float = 0.0

    # ---- closed-form solution ------------------------------------------
    def _k(self) -> float:
        return float(np.sqrt(max(self.c0, 1e-9) * max(self.c2, 1e-12)))

    def _A(self) -> float:
        return float(np.arctan(self.omega_ref * np.sqrt(self.c2 / self.c0)))

    def omega_at(self, t: float) -> float:
        """|angular speed| (deg/s) at absolute time t (s)."""
        arg = self._A() - self._k() * (t - self.t_ref)
        if arg <= 0:
            return 0.0
        return float(np.sqrt(self.c0 / self.c2) * np.tan(arg))

    def theta_at(self, t: float) -> float:
        """Unwrapped azimuth (deg) at absolute time t, including direction."""
        A, k = self._A(), self._k()
        arg = max(self._A() - k * (t - self.t_ref), 1e-9)
        span = (1.0 / self.c2) * (np.log(np.cos(arg)) - np.log(np.cos(A)))
        return float(self.theta_ref + self.direction * span)

    def time_to_speed(self, omega_target: float) -> float:
        """Absolute time (s) at which |speed| decays to omega_target."""
        A_target = np.arctan(omega_target * np.sqrt(self.c2 / self.c0))
        return float(self.t_ref + (self._A() - A_target) / self._k())

    # ---- fitting --------------------------------------------------------
    @classmethod
    def fit(cls, t: np.ndarray, theta_unwrapped: np.ndarray) -> "BallDecayModel":
        """Fit from unwrapped azimuth samples (deg) at times t (s).

        Robust to direction: works on |omega|. Requires >= ~0.5 s of data
        spanning a measurable speed change.
        """
        t = np.asarray(t, dtype=float)
        th = np.asarray(theta_unwrapped, dtype=float)
        omega_signed = smooth_derivative(t, th, window=min(11, len(t) | 1))
        direction = 1.0 if np.median(omega_signed) >= 0 else -1.0
        omega = np.abs(omega_signed)
        alpha = smooth_derivative(t, omega, window=min(11, len(t) | 1))
        # Linear regression: -alpha = c0 + c2 * omega^2
        # Trim edges where the derivative windows are one-sided.
        m = max(3, len(t) // 10)
        sl = slice(m, len(t) - m) if len(t) > 3 * m else slice(None)
        X = np.column_stack([np.ones_like(omega[sl]), omega[sl] ** 2])
        y = -alpha[sl]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        c0 = float(max(coef[0], 1e-3))
        c2 = float(max(coef[1], 1e-9))
        i_ref = (sl.start or 0)
        model = cls(
            c0=c0, c2=c2,
            t_ref=float(t[i_ref]),
            omega_ref=float(omega[i_ref]),
            theta_ref=float(th[i_ref]),
            direction=direction,
        )
        model._refine(t[sl], th[sl])
        pred = np.array([model.theta_at(tt) for tt in t[sl]])
        model.residual_rms = float(np.sqrt(np.mean((pred - th[sl]) ** 2)))
        return model

    # ------------------------------------------------------------------
    def _span(self, t: np.ndarray, c0: float, c2: float,
              omega_ref: float) -> np.ndarray:
        """Unsigned azimuth travelled since t_ref for given parameters."""
        k = np.sqrt(c0 * c2)
        A = np.arctan(omega_ref * np.sqrt(c2 / c0))
        arg = np.clip(A - k * (t - self.t_ref), 1e-6, None)
        return (np.log(np.cos(arg)) - np.log(np.cos(A))) / c2

    def _refine(self, t: np.ndarray, th: np.ndarray, iters: int = 40) -> None:
        """Levenberg-Marquardt on (log c0, log c2, omega_ref), fitting the
        analytic trajectory directly to the observed azimuths. theta_ref is
        solved in closed form at each step. This is far more accurate than
        the derivative regression, especially for extrapolation."""
        x = np.array([np.log(self.c0), np.log(self.c2), self.omega_ref])

        def residuals(x):
            # Real tracks (short arcs, tracker glitches) can drive the search
            # into overflow; clamp the log-parameters to a physical range so a
            # bad fit degrades into a large residual rather than inf/nan.
            lc0 = float(np.clip(x[0], -20.0, 20.0))
            lc2 = float(np.clip(x[1], -40.0, 5.0))
            c0, c2, om = np.exp(lc0), np.exp(lc2), max(x[2], 1.0)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                span = self._span(t, c0, c2, om)
            pred = self.direction * span
            if not np.all(np.isfinite(pred)):
                return np.full(len(t), 1e6), float(self.theta_ref)
            theta_ref = float(np.mean(th - pred))
            return (theta_ref + pred) - th, theta_ref

        lam = 1e-3
        r, theta_ref = residuals(x)
        cost = float(r @ r)
        for _ in range(iters):
            # Numerical Jacobian.
            J = np.empty((len(t), 3))
            for j in range(3):
                dx = np.zeros(3)
                dx[j] = 1e-5 * max(1.0, abs(x[j]))
                r2, _ = residuals(x + dx)
                J[:, j] = (r2 - r) / dx[j]
            H = J.T @ J + lam * np.eye(3)
            g = J.T @ r
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            x_new = x - step
            x_new[2] = max(x_new[2], 1.0)
            r_new, tref_new = residuals(x_new)
            cost_new = float(r_new @ r_new)
            if cost_new < cost:
                x, r, cost, theta_ref = x_new, r_new, cost_new, tref_new
                lam = max(lam * 0.3, 1e-8)
                if cost / len(t) < 1e-6:
                    break
            else:
                lam *= 10.0
                if lam > 1e6:
                    break
        self.c0 = float(np.exp(np.clip(x[0], -20.0, 20.0)))
        self.c2 = float(np.exp(np.clip(x[1], -40.0, 5.0)))
        self.omega_ref = float(max(x[2], 1.0))
        self.theta_ref = float(theta_ref)


@dataclass
class RotorModel:
    """Rotor azimuth: near-constant speed with slow linear deceleration."""

    phi0: float        # unwrapped azimuth (deg) at t0
    omega0: float      # signed speed deg/s at t0
    alpha: float       # signed acceleration deg/s^2 (small, <= 0 in magnitude)
    t0: float

    def phi_at(self, t: float) -> float:
        dt = t - self.t0
        # Don't let the fitted deceleration reverse the rotor direction.
        t_stop = -self.omega0 / self.alpha if self.alpha * self.omega0 < 0 else np.inf
        dt = min(dt, t_stop) if np.isfinite(t_stop) and t_stop > 0 else dt
        return float(self.phi0 + self.omega0 * dt + 0.5 * self.alpha * dt * dt)

    @classmethod
    def fit(cls, t: np.ndarray, phi_unwrapped: np.ndarray) -> "RotorModel":
        t = np.asarray(t, dtype=float)
        ph = np.asarray(phi_unwrapped, dtype=float)
        tm = t - t[0]
        coef = np.polyfit(tm, ph, 2)
        return cls(phi0=float(coef[2]), omega0=float(coef[1]),
                   alpha=float(2 * coef[0]), t0=float(t[0]))
