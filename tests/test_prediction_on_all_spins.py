"""The prediction, scored on every round in the video, leave-one-out.

For a long time this was measured on one spin, and one spin flattered it badly:
the error there was 0.38 s, and across the five rounds it is 0.62 s. Worse, the
single-spin number was not even stable — it moved to +1.06 s when the same spin
was rescanned with a different detection set, because the unwrapping was losing
laps. Both facts are pinned here so neither can quietly come back.

`tests/data/side_dets.npz` holds the side-view detections of all five rounds —
time, wheel azimuth, radius, confidence — already filtered to the ball track
and static-rejected, with each round's cutoff and the transfer instant measured
independently in the top-down view.

Every figure here is leave-one-out: the wheel calibration for each round is
fitted on the other four. With five spins that is the only kind of number worth
publishing, and it is what the README quotes.
"""

import numpy as np
import pytest

from prophetvision.earlyside import (OMEGA_TRANSFER_DEG_S, TransferCalibration,
                                     WheelDecay, fit_speed,
                                     fit_speed_circular)
from prophetvision.realstream import unwrap_predictive
from prophetvision.zone import wrapped_normal_coverage

DATA = __file__.rsplit("/", 1)[0] + "/data/side_dets.npz"
DECAY = WheelDecay(c0=17.5266174, c2=1.99170412e-4)
#: ball and rotor run opposite, so their rates add: pockets per second of
#: relative motion at the transfer, averaged over the five rounds.
RELATIVE_RATE = 18.32
#: measured spread from the transfer to the paid pocket (test_real_spins.py).
SIGMA_SCATTER = 7.21


@pytest.fixture(scope="module")
def rounds():
    z = np.load(DATA, allow_pickle=False)
    names = [str(n) for n in z["names"]]
    return {n: (z[f"side_{n}"], float(c), float(m))
            for n, c, m in zip(names, z["cutoff"], z["t_transfer"])}


def _speed_at_cutoff(rows, cutoff, fitter="circular"):
    t, az, conf = rows[:, 0], rows[:, 1], rows[:, 3]
    if fitter == "circular":
        _r, w0, _p = fit_speed_circular(t, az, DECAY, weights=conf)
    else:
        w0 = fit_speed(t, unwrap_predictive(t, az, direction=-1), DECAY)[1]
    return DECAY.speed_after(w0, cutoff - t[0])


@pytest.fixture(scope="module")
def fitted(rounds):
    names = list(rounds)
    w = np.array([_speed_at_cutoff(*rounds[n][:2]) for n in names])
    rem = np.array([rounds[n][2] - rounds[n][1] for n in names])
    return names, w, rem


def test_the_fitted_speed_is_not_the_weak_link(fitted):
    """It correlates 0.975 with the time the ball actually takes. What needed
    calibrating was the conversion from speed to time, not the speed."""
    _names, w, rem = fitted
    assert np.corrcoef(w, rem)[0, 1] > 0.95


def test_leave_one_out_prediction_error(fitted):
    names, w, rem = fitted
    errs = _loo_errors(w, rem)
    assert len(errs) == 5
    assert float(np.sqrt((errs ** 2).mean())) == pytest.approx(0.536, abs=0.03)
    assert np.abs(errs).max() < 1.0


def _loo_errors(w, rem):
    out = []
    for i in range(len(w)):
        tr = [j for j in range(len(w)) if j != i]
        cal = TransferCalibration.fit(w[tr], rem[tr], DECAY)
        out.append(cal.remaining(w[i]) - rem[i])
    return np.array(out)


def test_calibration_beats_the_uncalibrated_decay_law(fitted):
    """Uncalibrated the law is wrong in proportion to the extrapolation span —
    0.85 s rms. One fitted scale is the whole fix."""
    _names, w, rem = fitted
    raw = np.array([DECAY.time_to(x, OMEGA_TRANSFER_DEG_S) for x in w]) - rem
    rms_raw = float(np.sqrt((raw ** 2).mean()))
    rms_cal = float(np.sqrt((_loo_errors(w, rem) ** 2).mean()))
    assert rms_raw > 0.65
    assert rms_cal < rms_raw * 0.80


def test_the_calibrated_scale_is_stable_across_folds(fitted):
    """If it swung about, one round would be driving it and the leave-one-out
    number would mean nothing."""
    _names, w, rem = fitted
    scales = [TransferCalibration.fit(w[[j for j in range(5) if j != i]],
                                      rem[[j for j in range(5) if j != i]],
                                      DECAY).scale for i in range(5)]
    assert max(scales) - min(scales) < 0.05
    assert all(1.0 < s < 1.15 for s in scales)


def test_the_published_zone_coverage(fitted):
    """The headline. 18 chips fall short of 60%; 21 chips clear it. Both are
    quoted in the README, together with the chance floor, because a wider zone
    raises the floor too."""
    _names, w, rem = fitted
    rms = float(np.sqrt((_loo_errors(w, rem) ** 2).mean()))
    sigma = float(np.hypot(rms * RELATIVE_RATE, SIGMA_SCATTER))
    assert sigma == pytest.approx(12.18, abs=0.25)
    assert wrapped_normal_coverage(sigma, 18) == pytest.approx(0.564, abs=0.012)
    assert wrapped_normal_coverage(sigma, 21) == pytest.approx(0.645, abs=0.012)
    assert wrapped_normal_coverage(sigma, 21) > 0.60
    # the edge over betting the same number of chips blindly is what is real
    for width in (13, 18, 21, 24):
        edge = wrapped_normal_coverage(sigma, width) - width / 37
        assert 0.03 < edge < 0.12


def test_the_circular_fit_beats_the_unwrapped_one_on_real_data(rounds):
    """Not a style preference: the unwrapped fit loses whole laps at 500 deg/s,
    and the difference is why the earlier 18-chip figure had to be retracted."""
    names = list(rounds)
    rem = np.array([rounds[n][2] - rounds[n][1] for n in names])
    w_circ = np.array([_speed_at_cutoff(*rounds[n][:2], "circular")
                       for n in names])
    w_unwr = np.array([_speed_at_cutoff(*rounds[n][:2], "unwrapped")
                       for n in names])
    rms_c = float(np.sqrt((_loo_errors(w_circ, rem) ** 2).mean()))
    rms_u = float(np.sqrt((_loo_errors(w_unwr, rem) ** 2).mean()))
    assert rms_c < rms_u


# --- confidence gating: the system declines the rounds it cannot fit --------

@pytest.fixture(scope="module")
def gated(rounds):
    from prophetvision.earlyside import fit_speed_gated
    out = {}
    for n, (rows, cutoff, meas) in rounds.items():
        est = fit_speed_gated(rows[:, 0], rows[:, 1], DECAY, weights=rows[:, 3])
        w = DECAY.speed_after(est.omega0, cutoff - rows[0, 0])
        out[n] = (est, w, meas - cutoff)
    return out


def test_the_confidence_split_is_bimodal_not_a_tuned_threshold(gated):
    """Concentrations come out 0.44, 0.98, 0.98, 0.97, 0.44 — two clusters with
    nothing between them. Any threshold from 0.5 to 0.95 makes the same cut,
    which is what stops this being a knob fitted to the answer."""
    r = sorted(est.concentration for est, _w, _rem in gated.values())
    assert r[1] < 0.60 < 0.90 < r[2]
    accepted = [n for n, (est, _w, _rem) in gated.items() if est.trustworthy()]
    assert set(accepted) == {"spinA", "spinB", "spin4"}


def test_the_rounds_it_declines_are_the_ones_it_would_get_wrong(gated):
    """The gate has to correlate with accuracy or it is worthless. Accepted
    rounds land within 6% of the speed the outcome requires; declined ones are
    out by 15% and 23%."""
    need = {n: _needed_speed(rem) for n, (_e, _w, rem) in gated.items()}
    for n, (est, w, _rem) in gated.items():
        err = abs(w / need[n] - 1.0)
        assert (err < 0.07) == est.trustworthy(), (n, err)


def _needed_speed(remaining):
    import math
    s = math.sqrt(DECAY.c2 / DECAY.c0)
    k = math.sqrt(DECAY.c0 * DECAY.c2)
    return math.tan(math.atan(OMEGA_TRANSFER_DEG_S * s) + k * remaining) / s


def test_coverage_on_the_rounds_it_accepts(gated):
    """The headline of the gated pipeline: on the three rounds it will predict,
    18 chips cover 70% against a 48.6% floor — an edge of 21 points, against
    the 5 points the ungated pipeline gets by predicting everything.

    Three rounds. This is a hypothesis to test on more footage, not a
    demonstrated win rate, and the README says so."""
    acc = [n for n, (est, _w, _rem) in gated.items() if est.trustworthy()]
    w = np.array([gated[n][1] for n in acc])
    rem = np.array([gated[n][2] for n in acc])
    rms = float(np.sqrt((_loo_errors(w, rem) ** 2).mean()))
    assert rms == pytest.approx(0.324, abs=0.03)
    sigma = float(np.hypot(rms * RELATIVE_RATE, 5.18))
    assert sigma == pytest.approx(7.88, abs=0.2)
    assert wrapped_normal_coverage(sigma, 18) == pytest.approx(0.747, abs=0.015)
    assert wrapped_normal_coverage(sigma, 18) - 18 / 37 > 0.20


def test_gating_more_than_halves_the_prediction_error(gated):
    acc = [n for n, (est, _w, _rem) in gated.items() if est.trustworthy()]
    w_all = np.array([gated[n][1] for n in gated])
    rem_all = np.array([gated[n][2] for n in gated])
    w_acc = np.array([gated[n][1] for n in acc])
    rem_acc = np.array([gated[n][2] for n in acc])
    rms_all = float(np.sqrt((_loo_errors(w_all, rem_all) ** 2).mean()))
    rms_acc = float(np.sqrt((_loo_errors(w_acc, rem_acc) ** 2).mean()))
    assert rms_acc < rms_all / 2.0


# --- earlier emission: the user's constraint, measured -----------------------

def _loo_at(rounds, delta):
    """LOO timing errors with the cutoff shifted by ``delta`` seconds."""
    from prophetvision.earlyside import fit_speed_gated
    W, REM = {}, {}
    for n, (rows, cutoff, meas) in rounds.items():
        c = cutoff + delta
        s = rows[rows[:, 0] <= c]
        if len(s) < 12:
            continue
        est = fit_speed_gated(s[:, 0], s[:, 1], DECAY, weights=s[:, 3])
        W[n] = DECAY.speed_after(est.omega0, c - s[0, 0])
        REM[n] = meas - c
    names = list(W)
    errs = []
    for n in names:
        tr = [k for k in names if k != n]
        cal = TransferCalibration.fit([W[k] for k in tr],
                                      [REM[k] for k in tr], DECAY)
        errs.append(cal.remaining(W[n]) - REM[n])
    return np.array(errs)


def test_emitting_at_no_more_bets_costs_little(rounds):
    """Moving the emission 0.75 s earlier — to the 'no more bets' transition
    itself — keeps the timing error at the reference level. The prediction
    does not need the post-NMB data."""
    ref = float(np.sqrt((_loo_at(rounds, 0.0) ** 2).mean()))
    at_nmb = float(np.sqrt((_loo_at(rounds, -0.75) ** 2).mean()))
    assert at_nmb < ref + 0.25


def test_emitting_one_second_before_nmb_is_not_informative(rounds):
    """The UNGATED estimator at NMB-1, on this fixture: scatter ≈1.7 s ≈ 30
    pockets — uniform-level. This stays pinned as the floor the gate rescues
    the system from. The gated pipeline, with its calibration fitted on
    accepted rounds only, reaches 0.70 s rms at the same cutoff — see
    tests/test_pre_nmb_emission.py, which supersedes this number as the
    operational one."""
    errs = _loo_at(rounds, -1.75)
    scatter = float(errs.std(ddof=1))
    assert scatter > 1.2
    assert scatter * RELATIVE_RATE > 20.0     # pockets: uniform-level
