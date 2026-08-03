"""The published numbers, re-derived from the reference footage.

Everything upstream of these arrays needs the 1.5 GB video; everything
downstream is arithmetic. `tests/data/spins_rim.npz` holds the rim detections
of the five spins — time, image azimuth, radius as a fraction of the bowl,
detector confidence, and index in the rotor frame — so the impact measurement
and the win-rate claim in the README can be checked without it.

If a change to :mod:`prophetvision.impactmeas` moves any of these, the README
is wrong and must be updated with it.
"""

import numpy as np
import pytest

from prophetvision.earlyside import OMEGA_TRANSFER_DEG_S
from prophetvision.impactmeas import (bounce_spread, measure_impact,
                                      speed_at_radius)
from prophetvision.zone import coverage_ci, wrapped_normal_coverage

DATA = __file__.rsplit("/", 1)[0] + "/data/spins_rim.npz"
# Spin A, 72.0 s cutoff: the r=0.95 crossing predicted 0.38 s early, and the
# relative ball-rotor rate there is 16.5 pockets/s (see README).
SIGMA_PRED = 0.38 * 16.48


@pytest.fixture(scope="module")
def spins():
    z = np.load(DATA, allow_pickle=False)
    names = [str(n) for n in z["names"]]
    return {n: (z[f"rows_{n}"], float(r), int(res))
            for n, r, res in zip(names, z["rotor_pockets_s"], z["results"])}


def test_every_spin_yields_a_trustworthy_impact(spins):
    for name, (rows, rotor, _res) in spins.items():
        r = measure_impact(rows, rotor)
        assert r is not None, name
        assert r.trustworthy, (name, r)


def test_the_arcs_move_against_the_rotor_and_come_down(spins):
    """The two physical facts the measurement rests on. A fixed highlight
    satisfies neither, and every earlier version of this measurement was
    fooled by one."""
    for name, (rows, rotor, _res) in spins.items():
        r = measure_impact(rows, rotor)
        assert r.lab_rate_pockets_s * rotor < 0.0, name
        assert abs(r.lab_rate_pockets_s) > 3.0, name
        assert r.descent > 0.05, name


def test_impacts_match_the_published_values(spins):
    published = {"spin1": 15.95, "spinA": 29.09, "spinB": 2.16,
                 "spin4": 2.22, "spin5": 24.49}
    for name, (rows, rotor, _res) in spins.items():
        got = measure_impact(rows, rotor).index
        assert got == pytest.approx(published[name], abs=0.05), name


def test_the_transfer_speed_is_a_wheel_constant(spins):
    """The measurement the whole prediction now hangs on: the ball's speed at a
    fixed radius barely varies from spin to spin, unlike its speed or azimuth
    at the rim exit."""
    speeds = []
    for name, (rows, rotor, _res) in spins.items():
        s = speed_at_radius(rows, rotor, r_target=0.95)
        if s is None:
            assert name == "spinB", name    # its arc starts already at r=0.89
            continue
        speeds.append(abs(s[0]))
    speeds = np.array(speeds)
    assert len(speeds) == 4
    assert speeds.mean() == pytest.approx(OMEGA_TRANSFER_DEG_S, abs=1.0)
    assert speeds.std(ddof=1) / speeds.mean() < 0.02


def test_the_exit_is_not_a_wheel_constant(spins):
    """The contrast that justifies moving the target: where the ball leaves the
    rim scatters over more than a third of the wheel, so extrapolating to 'the
    rim exit' aims at something that is not there."""
    az = np.array([measure_impact(rows, rotor).azimuth
                   for rows, rotor, _ in spins.values()])
    spread = np.abs((az[:, None] - az[None, :] + 180) % 360 - 180).max()
    assert spread > 100.0


def test_bounce_spread_and_win_rate_match_the_readme(spins):
    """The published figures, end to end: spread from the predictable crossing
    to the paid pocket, and the 18-chip coverage that follows."""
    idx95, results = [], []
    for name, (rows, rotor, res) in spins.items():
        s = speed_at_radius(rows, rotor, r_target=0.95)
        if s is None:
            continue
        idx95.append(s[2])
        results.append(res)
    st = bounce_spread(idx95, results)
    assert st["n"] == 4
    assert st["sigma_pockets"] == pytest.approx(6.17, abs=0.05)

    est = coverage_ci(st["v"], SIGMA_PRED, width=18, seed=0, n_boot=6000)
    assert est.sigma_total == pytest.approx(8.79, abs=0.05)
    assert est.point == pytest.approx(0.696, abs=0.005)
    assert est.point > est.floor
    assert est.point == pytest.approx(
        wrapped_normal_coverage(est.sigma_total, 18), abs=1e-9)
    # ...and four spins cannot make that significant, whatever the point value
    assert est.rayleigh_p > 0.05
    assert not est.beats_chance


def test_the_rim_exit_anchor_is_reported_too(spins):
    """Anchoring at the rim exit instead gives a wider spread on five spins.
    It is a component, not an alternative total — there is no way to predict
    the exit instant — but the number is quoted in the README, so it is
    pinned here."""
    impacts, results = [], []
    for _name, (rows, rotor, res) in spins.items():
        r = measure_impact(rows, rotor)
        assert r.trustworthy
        impacts.append(r.index)
        results.append(res)
    st = bounce_spread(impacts, results)
    assert st["n"] == 5
    assert st["sigma_pockets"] == pytest.approx(9.62, abs=0.05)
