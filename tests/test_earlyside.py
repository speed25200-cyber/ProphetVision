import numpy as np
import pytest

from prophetvision.earlyside import (EarlyPrediction, RotorModel, WheelDecay,
                                     fit_speed, predict_early)


def test_two_point_calibration_is_exact_at_its_anchors():
    d = WheelDecay.from_two_decelerations(328.0, 38.9, 92.0, 19.2)
    assert d.c0 + d.c2 * 328.0 ** 2 == pytest.approx(38.9, abs=1e-6)
    assert d.c0 + d.c2 * 92.0 ** 2 == pytest.approx(19.2, abs=1e-6)
    assert d.c0 > 0 and d.c2 > 0


def test_roll_reaches_the_drop_speed_and_slows_monotonically():
    d = WheelDecay(c0=17.5, c2=2.0e-4)
    w, travel, t = d.roll(600.0, 0.0, omega_drop=55.0)
    assert w <= 55.0
    assert t > 0.0
    assert travel < 0.0                      # azimuth decreases
    # partial roll must leave a higher speed than the full one
    w_mid, _, _ = d.roll(600.0, 0.0, t1=t / 2)
    assert w_mid > w


def test_fit_speed_recovers_a_known_speed():
    d = WheelDecay(c0=17.5, c2=2.0e-4)
    true_w = 615.0
    t = np.arange(0.0, 1.5, 0.05)
    theta = []
    w, travel, tt = true_w, 0.0, 0.0
    for target in t:
        while tt < target - 1e-9:
            w -= (d.c0 + d.c2 * w * w) * 0.002
            travel -= w * 0.002
            tt += 0.002
        theta.append(travel)
    resid, w0, _ = fit_speed(np.asarray(t), np.asarray(theta) + 123.0, d)
    assert abs(w0 - true_w) < 8.0, w0
    assert resid < 1.0


def test_drop_time_is_highly_sensitive_to_speed():
    """The sensitivity that dominates the whole error budget: a 1% speed error
    must move the drop instant by roughly a tenth of a second."""
    d = WheelDecay(c0=17.5, c2=2.0e-4)
    _, _, t_ref = d.roll(473.0, 72.0, omega_drop=55.0)
    _, _, t_hi = d.roll(473.0 * 1.01, 72.0, omega_drop=55.0)
    shift = abs(t_hi - t_ref)
    assert 0.03 < shift < 0.25, shift


def test_predict_early_refuses_when_starved():
    d = WheelDecay(c0=17.5, c2=2.0e-4)
    rot = RotorModel(omega0=89.0, alpha=-1.2, t0=66.5, anchor_deg=0.0)
    dets = np.array([[70.0 + 0.1 * k, 10.0 * k % 360, 1.1, 0.9]
                     for k in range(5)])
    with pytest.raises(RuntimeError):
        predict_early(dets, cutoff=72.0, decay=d, rotor=rot)


def test_zone_and_coverage_are_consistent():
    p = EarlyPrediction(omega_at_cutoff=495.0, omega_sigma=0.025,
                        drop_time=86.3, index=9.5, sigma_pockets=11.6,
                        n_detections=19, fit_residual_deg=6.4,
                        samples=np.full(1000, 9.5))
    assert len(p.zone(9)) == 9
    assert p.coverage(9) == pytest.approx(1.0)   # all draws at the centre
    wide, narrow = p.coverage(21), p.coverage(3)
    assert wide >= narrow
