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
SIGMA_SCATTER = 6.55


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
    assert float(np.sqrt((errs ** 2).mean())) == pytest.approx(0.624, abs=0.02)
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
    assert rms_raw > 0.80
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
    assert sigma == pytest.approx(13.18, abs=0.15)
    assert wrapped_normal_coverage(sigma, 18) == pytest.approx(0.539, abs=0.008)
    assert wrapped_normal_coverage(sigma, 21) == pytest.approx(0.618, abs=0.008)
    assert wrapped_normal_coverage(sigma, 21) > 0.60
    # the edge over betting the same number of chips blindly is what is real
    for width in (13, 18, 21, 24):
        edge = wrapped_normal_coverage(sigma, width) - width / 37
        assert 0.03 < edge < 0.07


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
