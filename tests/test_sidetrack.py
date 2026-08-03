import numpy as np
import pytest

from prophetvision.sidetrack import (Ellipse, hough_trajectory, reject_static)


def test_ellipse_roundtrip():
    e = Ellipse(cx=613.0, cy=500.0, a=250.0, ratio=0.42, angle_deg=7.0)
    for az in (0.0, 37.0, 155.0, 289.0):
        for r in (0.8, 1.0, 1.25):
            x, y = e.to_image(r, az)
            az2, r2 = e.to_wheel(x, y)
            assert abs(((az2 - az + 180) % 360) - 180) < 1e-6
            assert abs(r2 - r) < 1e-9


def test_reject_static_drops_a_fixed_reflection():
    """A reflection pinned at one azimuth for seconds is not a ball."""
    t_ball = np.arange(0.0, 4.0, 0.05)
    az_ball = (-400.0 * t_ball) % 360.0
    t_static = np.arange(0.0, 4.0, 0.1)
    az_static = np.full_like(t_static, 139.0)
    t = np.r_[t_ball, t_static]
    az = np.r_[az_ball, az_static]
    keep = reject_static(t, az)
    assert keep[:len(t_ball)].mean() > 0.9      # ball survives
    assert keep[len(t_ball):].sum() == 0        # reflection removed


def test_hough_recovers_a_decaying_trajectory_through_gaps_and_clutter():
    rng = np.random.default_rng(4)
    omega0, alpha, phi = -520.0, 40.0, 332.0
    t = np.arange(0.0, 8.0, 0.05)
    # the ball is only visible on part of each revolution (near-side arc)
    az_true = (phi + omega0 * t + 0.5 * alpha * t * t) % 360.0
    visible = (az_true > 180.0) & (az_true < 320.0)
    t_ball, az_ball = t[visible], az_true[visible]
    az_ball = az_ball + rng.normal(0, 3.0, len(az_ball))
    # plus twice as much clutter at random azimuths
    t_clut = rng.uniform(0, 8, 2 * len(t_ball))
    az_clut = rng.uniform(0, 360, len(t_clut))
    tt = np.r_[t_ball, t_clut]
    aa = np.r_[az_ball, az_clut] % 360.0

    traj = hough_trajectory(tt, aa)
    assert abs(traj.omega_at(0.0) - omega0) < 60.0, traj.omega0
    # and it must extrapolate: speed several seconds later
    assert abs(traj.omega_at(8.0) - (omega0 + alpha * 8.0)) < 90.0
    assert traj.inliers.sum() >= 0.5 * len(t_ball)


def test_hough_needs_enough_detections():
    with pytest.raises(ValueError):
        hough_trajectory(np.arange(4.0), np.zeros(4))
