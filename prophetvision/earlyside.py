"""Landing-zone prediction from the side view, at an early cutoff.

The operational constraint is that the prediction must be issued **2-3 s before
the multiplier animation appears**, because after that the animation covers the
wheel. On the reference stream the animation starts at 74.00 s (spin A) and
122.00 s (spin B), so the cutoff is around 72 s — which leaves roughly 1.5-2 s
of ball data, since the ball only reaches the outer track at about 70.4 s.

That is far too little to identify the decay law, so the law is **not** fitted
here. `c0` and `c2` are properties of the wheel, calibrated once from a spin
where both a fast and a slow phase were observed, and only the ball's speed and
phase are fitted at prediction time. With coefficients fixed, one free speed
parameter is recoverable from ~19 detections spanning 1.5 s.

The error budget is dominated by one term, and it is worth stating because it
is not obvious. Near contact the ball and rotor turn at comparable, opposite
rates, so the *relative* angle — the only thing that fixes the pocket — moves at
about 160 deg/s, or 16.5 pockets per second. Every 0.1 s of error on the
predicted instant therefore costs about 1.6 pockets.

**What the extrapolation should aim at.** Aiming it at "the ball leaves the rim"
was the mistake. That is not a fixed speed: where the ball leaves depends on
which deflector it meets, and on the reference footage the exit speed varies
by 4% and the exit azimuth by 120 degrees. Aiming instead at the *transfer
speed* :data:`OMEGA_TRANSFER_DEG_S` — the speed at r = 0.95 of the bowl radius,
which is set by the bowl's geometry and measured at 93.6 +- 1.0 deg/s across
four spins — is well posed. On spin A with a 72.0 s cutoff the crossing is
predicted at 84.27 s against 84.65 s measured, an error of 0.38 s (bootstrap
spread 0.24 s), where the old target gave more than a second.

Everything downstream of that crossing — the rest of the descent, the deflector
and the bounce — is not predicted but *measured*, once, as a spread: 6.2 pockets
over four spins. See the README for the resulting zone coverage and for why four
spins cannot make it statistically significant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import WheelConfig
from .realstream import unwrap_predictive
from .sidetrack import reject_static


#: Speed at which the ball crosses r = 0.95 of the bowl radius, in deg/s.
#:
#: This is the target the extrapolation aims at, and getting it from a
#: measurement rather than a guess is what made the drop time predictable. The
#: earlier value, 55 deg/s, stood for "the ball leaves the rim" — an event that
#: is not a fixed speed at all, because it depends on which deflector the ball
#: happens to meet. The r = 0.95 crossing is different: the ball's radius is set
#: by gravity against the bowl slope, so the speed there is a property of the
#: wheel. Measured with :func:`prophetvision.impactmeas.speed_at_radius` on four
#: spins of the reference footage: 93.6 +- 1.0 deg/s, a spread of 1.0%. On spin
#: A the predicted crossing moved from 1.2 s of error to 0.38 s.
OMEGA_TRANSFER_DEG_S = 93.6


@dataclass
class WheelDecay:
    """Decay law of the ball on the rim: d(omega)/dt = -(c0 + c2 omega^2)."""

    c0: float
    c2: float

    @classmethod
    def from_two_decelerations(cls, omega_a: float, accel_a: float,
                               omega_b: float, accel_b: float) -> "WheelDecay":
        """Solve exactly from two (speed, deceleration) measurements.

        Least squares over noisy local estimates was biased on real data; two
        well-separated, well-measured points give a far better-conditioned
        answer (measured: 38.9 deg/s^2 at 328 deg/s and 19.2 at 92 deg/s).
        """
        c2 = (accel_a - accel_b) / (omega_a ** 2 - omega_b ** 2)
        c0 = accel_b - c2 * omega_b ** 2
        return cls(c0=float(max(c0, 1e-6)), c2=float(max(c2, 1e-12)))

    def roll(self, omega0: float, t0: float, t1: float | None = None,
             omega_drop: float = OMEGA_TRANSFER_DEG_S, dt: float = 0.002,
             max_span: float = 60.0):
        """Integrate forward. Returns (omega, travel_deg, t).

        Stops at ``t1`` if given, otherwise when the speed reaches
        ``omega_drop``, which defaults to the measured transfer speed
        :data:`OMEGA_TRANSFER_DEG_S` rather than a nominal rim-exit speed.
        """
        w = float(omega0)
        travel = 0.0
        t = float(t0)
        while True:
            if t1 is not None:
                if t >= t1:
                    break
            elif w <= omega_drop or t - t0 > max_span:
                break
            w -= (self.c0 + self.c2 * w * w) * dt
            travel -= w * dt          # ball azimuth decreases
            t += dt
            if w <= 0:
                break
        return w, travel, t


@dataclass
class RotorModel:
    """Rotor azimuth of the reference pocket, linear-deceleration model."""

    omega0: float
    alpha: float
    t0: float
    anchor_deg: float

    def azimuth_at(self, t) -> float:
        dt = np.asarray(t, dtype=float) - self.t0
        return float((self.anchor_deg + self.omega0 * dt
                      + 0.5 * self.alpha * dt * dt) % 360.0)


@dataclass
class EarlyPrediction:
    omega_at_cutoff: float
    omega_sigma: float
    drop_time: float
    index: float
    sigma_pockets: float
    n_detections: int
    fit_residual_deg: float
    samples: np.ndarray = field(repr=False, default=None)

    def zone(self, width: int, wheel: WheelConfig | None = None) -> list[int]:
        wheel = wheel or WheelConfig()
        n = wheel.n_pockets
        c = int(round(self.index)) % n
        half = width // 2
        return [wheel.pocket_order[(c + j) % n]
                for j in range(-half, width - half)]

    def coverage(self, width: int) -> float:
        """Fraction of the Monte-Carlo draws inside a zone of that width."""
        if self.samples is None:
            return float("nan")
        half = width // 2
        d = np.abs(((self.samples - self.index + 18.5) % 37) - 18.5)
        return float(np.mean(d <= half))


def robust_fit_speed(t, theta_unwrapped, decay: WheelDecay,
                     iterations: int = 4, clip_sigma: float = 2.5,
                     min_keep: int = 10):
    """Fit the speed, discarding detections the model cannot explain.

    On real footage a handful of detections land on a reflection or a
    neighbouring feature; left in, they inflated the fitted residual from 6 to
    29 degrees and the bootstrap spread on the speed from 0.9% to 1.7%. Since
    the landing error is about 4.2 pockets per 1% of speed error, that
    difference is worth several pockets, so the outliers are removed rather
    than absorbed.

    Returns ``(residual_deg, omega0, phase_offset, keep_mask)``.
    """
    t = np.asarray(t, dtype=float)
    th = np.asarray(theta_unwrapped, dtype=float)
    keep = np.ones(len(t), dtype=bool)
    resid = omega0 = offset = None
    for _ in range(iterations):
        resid, omega0, offset = fit_speed(t[keep], th[keep], decay)
        pred = _integrate_to(t, omega0, decay) + offset
        err = th - pred
        scale = float(np.std(err[keep]))
        new_keep = np.abs(err) < clip_sigma * max(scale, 1.0)
        if new_keep.sum() < min_keep or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    resid, omega0, offset = fit_speed(t[keep], th[keep], decay)
    return resid, omega0, offset, keep


def _integrate_to(times, omega0: float, decay: WheelDecay, dt: float = 0.002):
    """Cumulative azimuth travel at each requested time."""
    times = np.asarray(times, dtype=float)
    w, travel, tt = float(omega0), 0.0, float(times[0])
    out = []
    for target in times:
        while tt < target - 1e-9:
            w -= (decay.c0 + decay.c2 * w * w) * dt
            travel -= w * dt
            tt += dt
        out.append(travel)
    return np.asarray(out)


def fit_speed(t, theta_unwrapped, decay: WheelDecay,
              omega_grid=np.arange(300.0, 1200.0, 1.0)):
    """Fit the single free speed parameter with the decay law held fixed."""
    t = np.asarray(t, dtype=float)
    th = np.asarray(theta_unwrapped, dtype=float)
    best = None
    for w0 in omega_grid:
        w, travel, tt = w0, 0.0, t[0]
        pred = []
        dt = 0.002
        for target in t:
            while tt < target - 1e-9:
                w -= (decay.c0 + decay.c2 * w * w) * dt
                travel -= w * dt
                tt += dt
            pred.append(travel)
        pred = np.asarray(pred)
        offset = float(np.mean(th - pred))
        resid = float(np.sqrt(np.mean((offset + pred - th) ** 2)))
        if best is None or resid < best[0]:
            best = (resid, float(w0), offset)
    return best  # (residual_deg, omega0, phase_offset)


def predict_early(detections, cutoff: float, decay: WheelDecay,
                  rotor: RotorModel, r_band=(1.00, 1.40),
                  min_conf: float = 0.45, t_start: float | None = None,
                  omega_drop: float = OMEGA_TRANSFER_DEG_S,
                  n_boot: int = 120,
                  anchor_sigma_pockets: float = 1.3,
                  n_mc: int = 4000, seed: int = 0,
                  wheel: WheelConfig | None = None) -> EarlyPrediction:
    """Predict the landing index using only detections up to ``cutoff``.

    ``detections`` is an (N, 4+) array of [t, azimuth_deg, r, confidence] in the
    wheel frame (see :mod:`prophetvision.sidetrack` for the de-projection).
    """
    wheel = wheel or WheelConfig()
    sector = 360.0 / wheel.n_pockets
    d = np.asarray(detections, dtype=float)
    m = ((d[:, 2] >= r_band[0]) & (d[:, 2] < r_band[1])
         & (d[:, 3] > min_conf) & (d[:, 0] <= cutoff))
    if t_start is not None:
        m &= d[:, 0] >= t_start
    sub = d[m]
    if len(sub) < 10:
        raise RuntimeError(f"only {len(sub)} detections before the cutoff")
    sub = sub[np.argsort(sub[:, 0])]
    t, az = sub[:, 0], sub[:, 1]
    keep = reject_static(t, az)
    t, az = t[keep], az[keep]
    if len(t) < 10:
        raise RuntimeError(f"only {len(t)} detections after static rejection")
    th = unwrap_predictive(t, az, direction=-1)

    resid, omega0, offset, keep = robust_fit_speed(t, th, decay)
    t, th = t[keep], th[keep]

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(t), len(t))
        order = np.argsort(t[idx])
        tb, thb = t[idx][order], th[idx][order]
        if len(np.unique(tb)) < 8:
            continue
        boots.append(fit_speed(tb, thb, decay)[1])
    sigma_rel = float(np.std(boots) / np.mean(boots)) if len(boots) > 5 else 0.05

    def landing(w0, anchor_shift):
        w, tr1, _ = decay.roll(w0, t[0], t1=cutoff)
        _, tr2, td = decay.roll(w, cutoff, omega_drop=omega_drop)
        ball = (offset + tr1 + tr2) % 360.0
        zero = (rotor.azimuth_at(td) + anchor_shift) % 360.0
        return ((ball - zero) % 360.0) / sector, td

    index, drop = landing(omega0, 0.0)
    draws = np.array([landing(omega0 * (1 + rng.normal(0, sigma_rel)),
                              rng.normal(0, anchor_sigma_pockets * sector))[0]
                      for _ in range(n_mc)])
    ang = np.radians(draws * sector)
    r = abs(np.exp(1j * ang).mean())
    r = min(max(r, 1e-12), 1.0 - 1e-12)
    sigma = float(np.degrees(np.sqrt(-2.0 * np.log(r))) / sector)
    centre = float(np.degrees(np.arctan2(np.sin(ang).mean(),
                                         np.cos(ang).mean())) % 360.0 / sector)
    return EarlyPrediction(
        omega_at_cutoff=float(decay.roll(omega0, t[0], t1=cutoff)[0]),
        omega_sigma=sigma_rel, drop_time=float(drop), index=centre,
        sigma_pockets=sigma, n_detections=int(len(t)),
        fit_residual_deg=float(resid), samples=draws)
