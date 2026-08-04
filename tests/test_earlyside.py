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


def test_robust_fit_discards_outliers_and_tightens_the_speed():
    """Outlier detections must not be absorbed: at ~4.2 pockets per 1% of
    speed error, letting them in costs several pockets of landing accuracy."""
    from prophetvision.earlyside import robust_fit_speed, _integrate_to
    d = WheelDecay(c0=17.5, c2=2.0e-4)
    true_w = 615.0
    t = np.arange(0.0, 1.5, 0.03)
    th = _integrate_to(t, true_w, d) + 40.0
    rng = np.random.default_rng(0)
    th = th + rng.normal(0, 3.0, len(t))
    th[5] += 120.0            # detections that landed on something else
    th[17] -= 95.0
    th[26] += 140.0

    naive_resid, naive_w, _ = fit_speed(t, th, d)
    resid, w0, _, keep = robust_fit_speed(t, th, d)

    assert keep.sum() >= len(t) - 6
    assert not keep[[5, 17, 26]].any(), "the injected outliers were kept"
    assert resid < naive_resid / 2.0
    assert abs(w0 - true_w) < abs(naive_w - true_w) + 1e-9
    assert abs(w0 - true_w) < 12.0, w0


def test_closed_form_matches_the_integrator():
    """`roll` steps at 2 ms; the closed form is exact. They must agree, because
    the closed form is what the prediction now runs on."""
    d = WheelDecay(c0=17.5, c2=2.0e-4)
    w_end, travel, t_end = d.roll(600.0, 0.0, omega_drop=93.6)
    assert t_end == pytest.approx(d.time_to(600.0, 93.6), abs=0.01)
    assert d.speed_after(600.0, t_end) == pytest.approx(93.6, abs=0.5)
    assert float(d.travel(600.0, t_end)) == pytest.approx(travel, rel=2e-3)


def test_travel_broadcasts_over_a_speed_grid():
    d = WheelDecay(c0=17.5, c2=2.0e-4)
    grid = np.array([400.0, 600.0, 800.0])
    dt = np.array([0.0, 0.5, 1.0])
    out = d.travel(grid[:, None], dt[None, :])
    assert out.shape == (3, 3)
    assert np.allclose(out[:, 0], 0.0)
    assert np.all(np.diff(out[:, -1]) < 0)      # faster ball travels further


def test_circular_fit_survives_lap_ambiguity():
    """The bug this replaced: at 600 deg/s the ball laps every 0.6 s, and a
    predictive unwrap that misses one lap is wrong by 360 degrees for the rest
    of the window. The circular fit never unwraps, so it cannot."""
    from prophetvision.earlyside import fit_speed_circular
    d = WheelDecay(c0=17.5, c2=2.0e-4)
    true_w, phi = 615.0, 210.0
    t = np.concatenate([np.arange(0.0, 0.35, 0.02),      # gaps wide enough to
                        np.arange(1.10, 1.45, 0.02),     # hide whole laps
                        np.arange(2.20, 2.60, 0.02)])
    az = (phi + d.travel(true_w, t)) % 360.0
    rng = np.random.default_rng(0)
    az = (az + rng.normal(0, 2.0, len(t))) % 360.0
    r, w0, ph = fit_speed_circular(t, az, d)
    assert abs(w0 - true_w) < 8.0, w0
    assert abs((ph - phi + 180) % 360 - 180) < 8.0
    assert r > 0.9


def test_circular_fit_reports_a_low_score_on_noise():
    """The score has to be readable as a confidence, or a failed fit is
    indistinguishable from a good one — which is exactly how the unwrapped fit
    passed unnoticed with 231 degrees of residual."""
    from prophetvision.earlyside import fit_speed_circular
    d = WheelDecay(c0=17.5, c2=2.0e-4)
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 2.0, 60))
    az = rng.uniform(0, 360, 60)
    r, _w0, _ph = fit_speed_circular(t, az, d)
    assert r < 0.45
    good_t = np.arange(0.0, 2.0, 0.03)
    good_az = (100.0 + d.travel(615.0, good_t)) % 360.0
    assert fit_speed_circular(good_t, good_az, d)[0] > 0.95


def test_integrate_to_is_monotone_in_travel():
    from prophetvision.earlyside import _integrate_to
    d = WheelDecay(c0=17.5, c2=2.0e-4)
    t = np.linspace(0.0, 1.0, 20)
    tr = _integrate_to(t, 600.0, d)
    assert np.all(np.diff(tr) < 0)          # azimuth decreases throughout


def test_transfer_calibration_recovers_a_known_scale():
    """A wheel whose decay really is 6% slower than the coefficients say: the
    calibration has to find that, because on real footage it is exactly this
    kind of constant offset that the extrapolation turns into a second of
    error."""
    from prophetvision.earlyside import (OMEGA_TRANSFER_DEG_S,
                                         TransferCalibration)
    model = WheelDecay(c0=17.5, c2=2.0e-4)
    true = WheelDecay(c0=17.5 / 1.06, c2=2.0e-4 / 1.06)
    speeds = np.array([360.0, 440.0, 480.0, 520.0, 570.0])
    rem = np.array([true.time_to(w, OMEGA_TRANSFER_DEG_S) for w in speeds])
    cal = TransferCalibration.fit(speeds, rem, model)
    assert cal.scale == pytest.approx(1.06, rel=1e-3)
    assert cal.n_spins == 5
    for w, r in zip(speeds, rem):
        assert cal.remaining(w) == pytest.approx(r, abs=0.02)
    assert cal.transfer_time(72.0, 480.0) == pytest.approx(
        72.0 + cal.remaining(480.0))


def test_transfer_calibration_refuses_empty_input():
    from prophetvision.earlyside import TransferCalibration
    d = WheelDecay(c0=17.5, c2=2.0e-4)
    with pytest.raises(ValueError):
        TransferCalibration.fit([], [], d)
    with pytest.raises(ValueError):
        TransferCalibration.fit([400.0, 500.0], [12.0], d)


def test_an_uncalibrated_scale_is_the_identity():
    from prophetvision.earlyside import OMEGA_TRANSFER_DEG_S, TransferCalibration
    d = WheelDecay(c0=17.5, c2=2.0e-4)
    cal = TransferCalibration(decay=d)
    assert cal.remaining(500.0) == pytest.approx(
        d.time_to(500.0, OMEGA_TRANSFER_DEG_S))
