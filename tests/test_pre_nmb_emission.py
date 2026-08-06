"""Emission one second BEFORE "no more bets", at reference quality.

The earlier finding — that NMB-1 emission carries no information — was an
artifact of calibration hygiene, not physics. At that cutoff four of nine
rounds are unfittable (speed off by 21-48%), and the speed->time calibration
was being least-squares fitted ACROSS them, which corrupted it for the five
rounds that fit perfectly well. Fitting the calibration on the gate-accepted
rounds only — the same rounds a live system would bet on — recovers reference
quality one second before NMB. Measured on the same five scored rounds:
calibration on gated-only 0.70 s rms, calibration across all nine 1.14 s.
The gate must protect the calibration, not just the bet.

Live, no clairvoyance is needed: the system keeps a rolling estimate, and this
test shows the estimate that exists one second before the croupier's call is
already reference-grade.

`tests/data/prenmb_side.npz` freezes, per round of both videos: the side-view
detections (t, wheel azimuth, radius, confidence; band 0.90-1.50, conf >=
0.05, window NMB-3.2 to NMB+0.8), the NMB instant, the measured transfer
instant and rotor-frame index (from the top-down view, ellipse-independent),
the rotor rate and the paid pocket.

Same caveats as every number in this project: nine rounds, five accepted,
Rayleigh p = 0.52. Quality parity is demonstrated; statistical significance
is not, and cannot be at this sample size.
"""

import math

import numpy as np
import pytest

from prophetvision.earlyside import (EMISSION_DELTA_S, OMEGA_TRANSFER_DEG_S,
                                     TransferCalibration, WheelDecay, emit)
from prophetvision.impactmeas import bounce_spread
from prophetvision.zone import wrapped_normal_coverage

DATA = __file__.rsplit("/", 1)[0] + "/data/prenmb_side.npz"
DECAY = WheelDecay(c0=17.5266174, c2=1.99170412e-4)
POCKET = 360.0 / 37


@pytest.fixture(scope="module")
def rounds():
    z = np.load(DATA, allow_pickle=False)
    out = {}
    for i, n in enumerate([str(x) for x in z["names"]]):
        out[n] = dict(rows=z[f"side_{n}"], nmb=float(z["nmb"][i]),
                      t_meas=float(z["t_transfer"][i]),
                      idx_meas=float(z["idx_transfer"][i]),
                      rate=float(z["rotor_pockets_s"][i]),
                      result=int(z["results"][i]))
    return out


def _loo(rounds, delta, gated=True):
    W, REM, keep = {}, {}, []
    for n, r in rounds.items():
        em = emit(r["rows"], r["nmb"], DECAY, delta=delta)
        if em is None:
            continue
        W[n], REM[n] = em.omega_at_cutoff, r["t_meas"] - em.cutoff
        if em.played or not gated:
            keep.append(n)
    errs = {}
    for n in keep:
        tr = [k for k in keep if k != n]
        cal = TransferCalibration.fit([W[k] for k in tr],
                                      [REM[k] for k in tr], DECAY)
        errs[n] = cal.remaining(W[n]) - REM[n]
    return errs


def test_the_program_emits_one_second_before_nmb_by_default(rounds):
    """NMB-1 means one second BEFORE "no more bets", and it is the program's
    default, not a test-only setting: `emit(rows, nmb, decay)` with no delta
    cuts the data at nmb - 1.0 and reproduces the pinned operating point."""
    assert EMISSION_DELTA_S == -1.0
    r = rounds["spinA"]
    em = emit(r["rows"], r["nmb"], DECAY)
    assert em is not None
    assert em.cutoff == pytest.approx(r["nmb"] - 1.0)
    explicit = emit(r["rows"], r["nmb"], DECAY, delta=-1.0)
    assert em.omega_at_cutoff == explicit.omega_at_cutoff
    assert em.played == explicit.played


def test_gated_quality_at_nmb_minus_one_matches_the_reference(rounds):
    """The goal, verbatim: same quality, before NMB. Reference (NMB+0.75,
    gated): 0.71 s rms. At NMB-1.00, gated: 0.70 s."""
    ref = _loo(rounds, +0.75)
    early = _loo(rounds, -1.00)
    rms_ref = math.sqrt(np.mean(np.square(list(ref.values()))))
    rms_early = math.sqrt(np.mean(np.square(list(early.values()))))
    assert len(early) >= 5
    assert rms_early == pytest.approx(0.703, abs=0.05)
    assert rms_early <= rms_ref + 0.05


def test_the_gate_must_protect_the_calibration_not_just_the_bet(rounds):
    """The regression test for the actual insight. Same five rounds scored;
    only the calibration pool changes."""
    W, REM, gat = {}, {}, []
    for n, r in rounds.items():
        em = emit(r["rows"], r["nmb"], DECAY)
        if em is None:
            continue
        W[n], REM[n] = em.omega_at_cutoff, r["t_meas"] - em.cutoff
        if em.played:
            gat.append(n)

    def rms(pool):
        errs = []
        for n in gat:
            tr = [k for k in pool if k != n]
            cal = TransferCalibration.fit([W[k] for k in tr],
                                          [REM[k] for k in tr], DECAY)
            errs.append(cal.remaining(W[n]) - REM[n])
        return math.sqrt(np.mean(np.square(errs)))

    clean, dirty = rms(gat), rms(list(W))
    assert clean == pytest.approx(0.703, abs=0.05)
    assert dirty > clean + 0.30


def test_quality_holds_at_every_intermediate_cutoff(rounds):
    """No cliff between the reference and NMB-1: the gated rms stays within
    reference grade the whole way back. An emission policy can therefore pick
    any instant in that range."""
    for delta in (0.75, 0.0, -0.5, -1.0):
        errs = _loo(rounds, delta)
        assert len(errs) >= 4, delta
        rms = math.sqrt(np.mean(np.square(list(errs.values()))))
        assert rms < 1.0, (delta, rms)


def test_end_to_end_zone_at_nmb_minus_one(rounds):
    """From the pre-NMB emission all the way to the paid pocket."""
    errs = _loo(rounds, -1.00)
    ip, res = [], []
    for n, e in errs.items():
        r = rounds[n]
        rel = -(OMEGA_TRANSFER_DEG_S / POCKET) - r["rate"]
        ip.append((r["idx_meas"] + e * rel) % 37.0)
        res.append(r["result"])
    st = bounce_spread(ip, res)
    assert st["n"] == 5
    assert st["sigma_pockets"] == pytest.approx(8.26, abs=0.35)
    cov = wrapped_normal_coverage(st["sigma_pockets"], 18)
    assert cov == pytest.approx(0.725, abs=0.02)
    assert cov - 18 / 37 > 0.20
    # and, as everywhere in this project at this sample size:
    assert st["rayleigh_p"] > 0.05


def test_the_abstention_rate_is_reported_not_hidden(rounds):
    """Five of nine rounds are played, four declined. The rate is part of the
    result: a system that abstains often needs more rounds to realise its
    edge, and hiding that would flatter the headline."""
    all_fits = _loo(rounds, -1.00, gated=False)
    gated = _loo(rounds, -1.00, gated=True)
    assert len(all_fits) == 9
    assert len(gated) == 5


def test_the_cost_curve_back_to_nmb_minus_two(rounds):
    """How far back can the emission go? The timing rms holds reference grade
    (< 0.8 s) all the way to NMB-2 on the rounds that still fit — but the
    number of rounds that fit at all falls from 9 to 4, because the ball is
    only launched around NMB-3 and at NMB-2 the measurable arc is under a
    second. The degradation mode is coverage, not accuracy: fewer playable
    rounds, not worse predictions on the playable ones. No ball-based
    prediction can exist before the ball does, so NMB-3 is the hard wall."""
    fitted_at = {}
    for delta in (-1.0, -1.25, -1.5, -2.0):
        errs = _loo(rounds, delta)
        fitted_at[delta] = len(errs)
        if len(errs) >= 3:
            rms = math.sqrt(np.mean(np.square(list(errs.values()))))
            assert rms < 0.80, (delta, rms)
    assert fitted_at[-1.0] == 5
    assert fitted_at[-2.0] <= 4        # coverage is what collapses
