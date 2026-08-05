"""Video 2: the calibration from video 1, applied blind to new footage.

Nothing here was fitted on video 2. The decay coefficients, the transfer speed,
the wheel scale (1.0584), the side-view ellipse and the abstention threshold
all come from video 1. Video 2 supplied only its own camera cuts and its own
"no more bets" timing, neither of which involves an outcome.

`tests/data/video2_rounds.npz` freezes the detections of the six rounds so this
runs without the video: for each round, the top-down table (t, azimuth, radius,
confidence, rotor-frame index), the side-view rows, the rotor rate, the cutoff
and the paid pocket.

Two things are pinned here that are *not* good news, because they are what the
test actually showed:

* only four of six rounds can be measured at all -- on the other two the
  top-down view appears after the ball has already descended, so there is no
  arc to anchor the reference on;
* the abstention gate accepted all four it could measure, so on this footage it
  did no filtering. Its discriminating power, the weakest claim from video 1,
  is untested rather than confirmed.
"""

import numpy as np
import pytest

from prophetvision.earlyside import (OMEGA_TRANSFER_DEG_S, TransferCalibration,
                                     WheelDecay, fit_speed_gated)
from prophetvision.impactmeas import bounce_spread, transfer_point
from prophetvision.zone import wrapped_normal_coverage

DATA = __file__.rsplit("/", 1)[0] + "/data/video2_rounds.npz"
DECAY = WheelDecay(c0=17.5266174, c2=1.99170412e-4)
#: fitted on video 1's five rounds, leave-one-out. Frozen.
CAL = TransferCalibration(decay=DECAY, scale=1.0584,
                          omega_transfer=OMEGA_TRANSFER_DEG_S, n_spins=5)
POCKET = 360.0 / 37


@pytest.fixture(scope="module")
def rounds():
    z = np.load(DATA, allow_pickle=False)
    out = {}
    for n, rate, cut, res in zip([str(x) for x in z["names"]],
                                 z["rotor_pockets_s"], z["cutoff"],
                                 z["results"]):
        out[n] = {"td": z[f"td_{n}"], "side": z[f"side_{n}"],
                  "rate": float(rate), "cutoff": float(cut), "result": int(res)}
    return out


def _predicted_index(r):
    """Rotor-frame index at the predicted transfer instant, or None."""
    tp = transfer_point(r["td"], r["rate"], DECAY, OMEGA_TRANSFER_DEG_S)
    if tp is None:
        return None
    t_meas, idx_meas = tp
    s = r["side"]
    est = fit_speed_gated(s[:, 0], s[:, 1], DECAY, weights=s[:, 3])
    w = DECAY.speed_after(est.omega0, r["cutoff"] - s[0, 0])
    t_pred = CAL.transfer_time(r["cutoff"], w)
    # ball and rotor close on each other; the anchor t_meas cancels because
    # (t_meas, idx_meas) is just a point on the measured trajectory.
    rel = -(OMEGA_TRANSFER_DEG_S / POCKET) - r["rate"]
    return (idx_meas + (t_pred - t_meas) * rel) % 37.0, est


def test_two_rounds_cannot_be_measured_at_all(rounds):
    """Not a prediction failure -- a coverage failure. On these two the
    top-down cut lands after the ball has come down, so there is no rim arc to
    put the reference on."""
    missing = [n for n, r in rounds.items()
               if transfer_point(r["td"], r["rate"], DECAY,
                                 OMEGA_TRANSFER_DEG_S) is None]
    assert len(missing) == 2


def test_the_gate_accepted_every_round_it_could_measure(rounds):
    """On video 1 the concentration split 0.44 / 0.98. Here every round scores
    high, so the gate filtered nothing and its value is untested."""
    scores = []
    for _n, r in rounds.items():
        out = _predicted_index(r)
        if out is None:
            continue
        scores.append(out[1])
    assert len(scores) == 4
    assert all(e.trustworthy() for e in scores)
    assert min(e.concentration for e in scores) > 0.90


def test_end_to_end_spread_on_new_footage(rounds):
    """The number that matters: how far the paid pocket sits from the predicted
    one, over rounds the calibration never saw."""
    idx, res = [], []
    for _n, r in rounds.items():
        out = _predicted_index(r)
        if out is None:
            continue
        idx.append(out[0])
        res.append(r["result"])
    st = bounce_spread(idx, res)
    assert st["n"] == 4
    assert st["sigma_pockets"] == pytest.approx(5.44, abs=0.3)
    assert wrapped_normal_coverage(st["sigma_pockets"], 18) > 0.60
    # ...and four rounds cannot make it significant
    assert st["rayleigh_p"] > 0.05
