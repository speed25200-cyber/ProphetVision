import math

import numpy as np
import pytest

from prophetvision.zone import (circular_sigma, coverage_ci,
                                sigma_for_coverage, wrapped_normal_coverage)


def test_a_useless_prediction_falls_back_to_the_chance_floor():
    """18 chips already cover 48.6% of a 37-pocket wheel; a zone is only worth
    playing insofar as it beats that."""
    assert wrapped_normal_coverage(1e6, 18) == pytest.approx(18 / 37, abs=0.01)
    assert wrapped_normal_coverage(0.01, 18) == pytest.approx(1.0)


def test_wrapping_is_not_ignored():
    """At the spreads this project reaches, the tails come back round the wheel
    and a plain normal understates the coverage by several points."""
    sigma = 11.0
    plain = math.erf(9.0 / (sigma * math.sqrt(2.0)))
    wrapped = wrapped_normal_coverage(sigma, 18)
    assert wrapped > plain + 0.01


def test_coverage_grows_with_the_zone_and_shrinks_with_the_spread():
    assert wrapped_normal_coverage(6.0, 24) > wrapped_normal_coverage(6.0, 12)
    assert wrapped_normal_coverage(4.0, 18) > wrapped_normal_coverage(9.0, 18)
    assert wrapped_normal_coverage(6.0, 37) == 1.0


def test_sigma_for_coverage_inverts_the_coverage():
    for target in (0.55, 0.65, 0.75):
        s = sigma_for_coverage(target, 18)
        assert wrapped_normal_coverage(s, 18) == pytest.approx(target, abs=1e-3)


def test_sigma_for_a_target_below_the_floor_is_unbounded():
    assert sigma_for_coverage(0.40, 18) == float("inf")


def test_the_60_70_target_sets_a_concrete_budget():
    """What the user asked for, translated: 18 chips at 60-70% is a statement
    about the total spread, and this is the number the pipeline must hit."""
    lo = sigma_for_coverage(0.70, 18)
    hi = sigma_for_coverage(0.60, 18)
    assert 5.0 < lo < 9.0, lo
    assert 8.0 < hi < 14.0, hi
    assert lo < hi


def test_circular_sigma_ignores_a_constant_offset():
    d = np.array([2.0, -1.0, 0.5, 3.0, -2.5])
    assert circular_sigma(d + 11.0) == pytest.approx(circular_sigma(d), abs=1e-9)


def test_circular_sigma_handles_the_seam():
    tight = np.array([36.5, 0.2, 36.8, 0.5, 0.0])
    assert circular_sigma(tight) < 1.0


def test_coverage_ci_widens_with_fewer_spins():
    rng = np.random.default_rng(1)
    big = rng.normal(0.0, 5.0, 40)
    small = big[:5]
    wide = coverage_ci(small, 4.4, n_boot=800, seed=1)
    narrow = coverage_ci(big, 4.4, n_boot=800, seed=1)
    assert (wide.hi - wide.lo) > (narrow.hi - narrow.lo)
    assert wide.n_spins == 5 and narrow.n_spins == 40


def test_coverage_ci_brackets_its_own_point_estimate():
    rng = np.random.default_rng(2)
    dev = rng.normal(0.0, 4.0, 12)
    est = coverage_ci(dev, 4.0, n_boot=1500, seed=2)
    assert est.lo <= est.point <= est.hi
    assert est.sigma_lo <= est.sigma_total <= est.sigma_hi
    assert est.floor == pytest.approx(18 / 37)


def test_a_scattered_bounce_cannot_beat_chance():
    """With six spins the bootstrap alone says the zone wins 51% of the time —
    resampling duplicates makes uniform data look concentrated. The Rayleigh
    gate is what stops that claim."""
    rng = np.random.default_rng(5)
    dev = rng.uniform(0, 37, 6)
    est = coverage_ci(dev, 4.4, n_boot=800, seed=5)
    assert est.rayleigh_p > 0.05
    assert not est.beats_chance


def test_a_tight_bounce_beats_chance_convincingly():
    rng = np.random.default_rng(7)
    dev = rng.normal(0.0, 2.0, 8)
    est = coverage_ci(dev, 4.4, n_boot=800, seed=7)
    assert est.beats_chance
    assert est.point > 0.75


def test_coverage_ci_refuses_a_single_spin():
    with pytest.raises(ValueError):
        coverage_ci([3.0], 4.4)
