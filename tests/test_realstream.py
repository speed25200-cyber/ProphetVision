import numpy as np

from prophetvision.realstream import (YellowBallTracker, ZeroMarkerTracker,
                                      predict_spin, unwrap_directional)
from prophetvision.synthetic import SyntheticSpin


def test_unwrap_directional_survives_gaps():
    # Ball at -100 deg/s sampled with a 1.7 s hole: nearest-wrap would flip
    # the direction on the gap step; directional unwrap must not.
    t = np.r_[np.arange(0, 1.0, 0.05), np.arange(2.7, 3.5, 0.05)]
    true = (360.0 - 100.0 * t) % 360.0
    u = unwrap_directional(true, direction=-1)
    assert np.all(np.diff(u) <= 1e-9)
    assert abs((u[-1] - u[0]) - (-100.0 * (t[-1] - t[0]))) < 1.0


def test_real_pipeline_on_synthetic_spin():
    spin = SyntheticSpin(seed=5)
    frames, times, truth = spin.simulate()
    cx = cy = spin.size / 2.0
    R = spin.size * 0.45

    ball = YellowBallTracker(cx, cy, R, r_inner=0.80, r_outer=1.05,
                             diff_threshold=12.0, min_score=100.0)
    rotor = ZeroMarkerTracker(cx, cy, R, ring_r=0.55, ring_width=0.10)
    cutoff = truth.t_drop - 1.5
    for f, t in zip(frames, times):
        if t > cutoff:
            break
        ball.feed(f, t)
        rotor.feed(f, t)

    samples = ball.samples()
    assert len(samples) > 60, f"too few ball samples: {len(samples)}"
    pred = predict_spin(samples, rotor, cutoff=cutoff, direction=+1)
    # The synthetic ball spins at +; rotor speed must match the generator.
    assert abs(abs(pred.rotor_speed) - abs(spin.rotor_omega)) < 8.0
    # Drop-time prediction within half a second, 1.5 s ahead.
    t_drop_pred = pred.model.time_to_speed(spin.omega_drop_true)
    assert abs(t_drop_pred - truth.t_drop) < 0.5
