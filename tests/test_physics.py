import numpy as np

from prophetvision.physics import BallDecayModel, RotorModel


def simulate_theta(omega0, c0, c2, t):
    """Numerically integrate the decay ODE for ground truth."""
    dt = 1e-4
    theta, omega, out, ti = 0.0, omega0, [], 0.0
    for target in t:
        while ti < target:
            theta += omega * dt
            omega -= (c0 + c2 * omega * omega) * dt
            ti += dt
        out.append(theta)
    return np.array(out), omega


def test_ball_decay_fit_and_extrapolation():
    c0, c2, omega0 = 170.0, 6e-5, 1900.0
    t_obs = np.arange(0, 2.0, 1 / 60)
    theta_obs, _ = simulate_theta(omega0, c0, c2, t_obs)
    rng = np.random.default_rng(0)
    noisy = theta_obs + rng.normal(0, 0.3, len(theta_obs))

    model = BallDecayModel.fit(t_obs, noisy)
    # Coefficients recovered within a reasonable factor.
    assert 0.5 * c0 < model.c0 < 2.0 * c0
    # Extrapolate 3 s past the observation window; check azimuth error.
    t_future = np.array([3.0, 4.0])
    theta_future, _ = simulate_theta(omega0, c0, c2, t_future)
    for tf, th_true in zip(t_future, theta_future):
        assert abs(model.theta_at(tf) - th_true) < 30.0  # deg


def test_time_to_speed_matches_integration():
    c0, c2, omega0 = 170.0, 6e-5, 1900.0
    t_obs = np.arange(0, 2.0, 1 / 60)
    theta_obs, _ = simulate_theta(omega0, c0, c2, t_obs)
    model = BallDecayModel.fit(t_obs, theta_obs)

    # Integrate until omega hits 640 deg/s for the true crossing time.
    dt, omega, ti = 1e-4, omega0, 0.0
    while omega > 640.0:
        omega -= (c0 + c2 * omega * omega) * dt
        ti += dt
    assert abs(model.time_to_speed(640.0) - ti) < 0.15  # s


def test_rotor_fit():
    t = np.arange(0, 3, 1 / 60)
    phi = 10.0 - 110.0 * t + 0.5 * 1.5 * t * t
    m = RotorModel.fit(t, phi)
    assert abs(m.omega0 - (-110.0)) < 1.0
    assert abs(m.phi_at(4.0) - (10.0 - 110.0 * 4 + 0.75 * 16)) < 2.0
