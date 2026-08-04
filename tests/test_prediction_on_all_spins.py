"""The prediction, scored on every round it can be scored on.

For a long time this was measured on one spin, and one spin flattered it badly:
the error there was 0.38 s, and across four rounds it is 0.73 s rms. Worse, the
single-spin number was not even stable — it moved to +1.06 s when the same spin
was rescanned with a different detection set, because the unwrapping was losing
laps. Both facts are pinned here so neither can quietly come back.

`tests/data/side_dets.npz` holds the side-view detections of the four rounds
that have them — time, wheel azimuth, radius, confidence — already filtered to
the ball track and static-rejected, together with each round's cutoff and the
transfer instant measured independently in the top-down view.
"""

import numpy as np
import pytest

from prophetvision.earlyside import (OMEGA_TRANSFER_DEG_S, WheelDecay,
                                     fit_speed, fit_speed_circular)
from prophetvision.realstream import unwrap_predictive

DATA = __file__.rsplit("/", 1)[0] + "/data/side_dets.npz"
DECAY = WheelDecay(c0=17.5266174, c2=1.99170412e-4)
#: ball and rotor run opposite, so their rates add: pockets per second of
#: relative motion at the transfer instant, averaged over the five spins.
RELATIVE_RATE = 18.32


@pytest.fixture(scope="module")
def rounds():
    z = np.load(DATA, allow_pickle=False)
    names = [str(n) for n in z["names"]]
    return {n: (z[f"side_{n}"], float(c), float(m))
            for n, c, m in zip(names, z["cutoff"], z["t_transfer"])}


def _predict(rows, cutoff, fitter):
    t, az, conf = rows[:, 0], rows[:, 1], rows[:, 3]
    if fitter == "circular":
        _r, w0, _p = fit_speed_circular(t, az, DECAY, weights=conf)
    else:
        th = unwrap_predictive(t, az, direction=-1)
        w0 = fit_speed(t, th, DECAY)[1]
    wc = DECAY.speed_after(w0, cutoff - t[0])
    return cutoff + DECAY.time_to(wc, OMEGA_TRANSFER_DEG_S)


def test_every_round_is_predicted_within_a_second_and_a_half(rounds):
    for name, (rows, cutoff, meas) in rounds.items():
        err = _predict(rows, cutoff, "circular") - meas
        assert abs(err) < 1.5, (name, err)


def test_the_published_prediction_error_holds(rounds):
    errs = np.array([_predict(rows, cutoff, "circular") - meas
                     for rows, cutoff, meas in rounds.values()])
    rms = float(np.sqrt((errs ** 2).mean()))
    assert len(errs) == 4
    assert rms == pytest.approx(0.734, abs=0.02)
    assert rms * RELATIVE_RATE == pytest.approx(13.45, abs=0.4)


def test_the_circular_fit_beats_the_unwrapped_one_on_real_data(rounds):
    """Not a style preference: the unwrapped fit is worse by more than a factor
    of two on the reference footage, and the difference is the whole reason the
    18-chip figure had to be retracted."""
    circ = np.array([_predict(r, c, "circular") - m
                     for r, c, m in rounds.values()])
    unwr = np.array([_predict(r, c, "unwrapped") - m
                     for r, c, m in rounds.values()])
    rms_c = float(np.sqrt((circ ** 2).mean()))
    rms_u = float(np.sqrt((unwr ** 2).mean()))
    assert rms_c < rms_u / 2.0, (rms_c, rms_u)


def test_the_prediction_alone_busts_the_whole_18_chip_budget(rounds):
    """The honest negative, stated where it bites. An 18-chip zone at 70% needs
    a total spread of 8.7 pockets for everything — prediction, descent, bounce.
    The prediction term alone is 13.4, so no zone of that width can reach the
    target however well the rest behaves. Half the rounds do clear the 0.39 s a
    13-pocket zone would need, which is why this is a budget failure and not a
    broken method."""
    from prophetvision.zone import sigma_for_coverage
    errs = np.array([_predict(r, c, "circular") - m
                     for r, c, m in rounds.values()])
    sigma_pred = float(np.sqrt((errs ** 2).mean())) * RELATIVE_RATE
    assert sigma_pred > sigma_for_coverage(0.70, 18)
    assert sum(abs(errs) < 0.39) <= len(errs) // 2
