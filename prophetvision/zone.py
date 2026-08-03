"""From an error budget to a playable zone, and to the rate it would win at.

The question the project is finally judged on is not "how many pockets is the
error" but "if I cover 18 of the 37 pockets, how often is the ball in them".
Those are not the same question, because 18 chips already cover 48.6% of the
wheel by luck alone. A zone is only worth playing if it beats that floor, and
by enough to survive the fact that the floor is measured on a handful of spins.

Two helpers here, deliberately separate:

* :func:`wrapped_normal_coverage` — the analytic answer for a given total
  spread. It sums the wrapped tails rather than truncating, because at the
  spreads this project actually achieves (5-12 pockets on a 37-pocket wheel)
  the wrap contributes several percent and ignoring it flatters the result.
* :func:`coverage_ci` — the same number with the sampling uncertainty of a
  small spin count carried through by bootstrap. With five spins the interval
  on the win rate is tens of points wide, which is the honest headline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def wrapped_normal_coverage(sigma_pockets: float, width: int,
                            n_pockets: int = 37, wraps: int = 6) -> float:
    """P(the ball lands in a contiguous zone of ``width`` pockets).

    The zone is centred on the prediction; the error is wrapped normal with
    ``sigma_pockets``. ``width`` counts pockets, so the half-width is
    ``width / 2`` — a zone of 18 reaches 9 pockets each way.

    Two series are used because neither converges everywhere: summing the
    shifted Gaussians works while the spread is small compared with the wheel,
    and collapses once it is not (a spread of a thousand pockets would need a
    thousand wraps); the Fourier form is the reverse, and it is the one that
    correctly returns the ``width / n_pockets`` chance floor in the limit.
    """
    if width >= n_pockets:
        return 1.0
    if sigma_pockets <= 0:
        return 1.0
    half = width / 2.0
    if sigma_pockets < n_pockets / 4.0:
        total = 0.0
        for k in range(-wraps, wraps + 1):
            hi = (half + k * n_pockets) / (sigma_pockets * math.sqrt(2.0))
            lo = (-half + k * n_pockets) / (sigma_pockets * math.sqrt(2.0))
            total += 0.5 * (math.erf(hi) - math.erf(lo))
        return float(min(max(total, 0.0), 1.0))
    total = width / n_pockets
    for k in range(1, 64):
        damp = math.exp(-0.5 * (sigma_pockets * 2.0 * math.pi * k
                                / n_pockets) ** 2)
        if damp < 1e-15:
            break
        total += (2.0 / (math.pi * k)) * damp * math.sin(
            2.0 * math.pi * k * half / n_pockets)
    return float(min(max(total, 0.0), 1.0))


def sigma_for_coverage(target: float, width: int, n_pockets: int = 37,
                       lo: float = 0.05, hi: float = 200.0) -> float:
    """Largest spread that still reaches ``target`` coverage with ``width``.

    Returns inf when the target is at or below the chance floor (width /
    n_pockets), because then any spread will do, and nan when it is
    unreachable.
    """
    if target <= width / n_pockets:
        return float("inf")
    if wrapped_normal_coverage(lo, width, n_pockets) < target:
        return float("nan")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if wrapped_normal_coverage(mid, width, n_pockets) >= target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def circular_sigma(deviations, n_pockets: int = 37) -> float:
    """Circular standard deviation of pocket deviations, in pockets."""
    ang = np.radians(np.asarray(deviations, dtype=float) * (360.0 / n_pockets))
    r = abs(np.exp(1j * ang).mean())
    r = min(max(r, 1e-12), 1.0 - 1e-12)
    return float(np.degrees(math.sqrt(-2.0 * math.log(r))) / (360.0 / n_pockets))


@dataclass
class CoverageEstimate:
    width: int
    point: float
    lo: float
    hi: float
    sigma_total: float
    sigma_lo: float
    sigma_hi: float
    n_spins: int
    floor: float
    rayleigh_p: float = float("nan")

    @property
    def beats_chance(self) -> bool:
        """True only if the zone beats luck on both counts that matter.

        The bootstrap interval alone is not enough. Resampling five or six
        points with replacement produces duplicate-heavy draws that look far
        more concentrated than the data are, so a scattered bounce can still
        show a lower bound above the floor. The Rayleigh test asks the question
        directly — are these deviations more concentrated than uniform — and it
        does not have that failure mode at small counts.
        """
        return self.lo > self.floor and self.rayleigh_p < 0.05


def coverage_ci(bounce_deviations, sigma_pred_pockets: float, width: int = 18,
                n_pockets: int = 37, n_boot: int = 4000, level: float = 0.90,
                seed: int = 0) -> CoverageEstimate:
    """Win rate of a ``width``-pocket zone, with a bootstrap interval.

    ``bounce_deviations`` are the per-spin ``result_index - impact_index``
    values (their *spread* is what matters; the mean is absorbed by the
    unknown numbering offset). ``sigma_pred_pockets`` is the spread the
    upstream prediction contributes on top of the bounce.
    """
    dev = np.asarray(bounce_deviations, dtype=float)
    if len(dev) < 2:
        raise ValueError("need at least two spins to estimate a spread")
    rng = np.random.default_rng(seed)

    def total(d):
        return math.hypot(circular_sigma(d, n_pockets), sigma_pred_pockets)

    s_hat = total(dev)
    boots = np.array([total(rng.choice(dev, len(dev), replace=True))
                      for _ in range(n_boot)])
    cov = np.array([wrapped_normal_coverage(s, width, n_pockets)
                    for s in boots])
    ang = np.radians(dev * (360.0 / n_pockets))
    m = len(dev)
    r = float(abs(np.exp(1j * ang).mean()))
    z = m * r * r
    p = float(np.exp(-z) * (1.0 + (2.0 * z - z * z) / (4.0 * m)))

    a = (1.0 - level) / 2.0
    return CoverageEstimate(
        width=width,
        point=wrapped_normal_coverage(s_hat, width, n_pockets),
        lo=float(np.quantile(cov, a)), hi=float(np.quantile(cov, 1.0 - a)),
        sigma_total=s_hat,
        sigma_lo=float(np.quantile(boots, a)),
        sigma_hi=float(np.quantile(boots, 1.0 - a)),
        n_spins=m, floor=width / n_pockets,
        rayleigh_p=float(min(max(p, 0.0), 1.0)))
