"""Track-before-detect must find a ball that per-frame detection misses.

The oblique synthetic render reproduces the regime that defeated every
blob-based tracker in this project: the wheel is foreshortened, the frame is
dimmed and noised, and the ball is below threshold in any single frame. The
velocity-matched filter integrates along the trajectory instead, so it should
still recover the ball's angular velocity.
"""

import numpy as np

from prophetvision.synthetic import SyntheticSpin
from prophetvision.vmf import (SpaceTime, build_spacetime, detector_power,
                               measure_rotor, null_rotor,
                               velocity_matched_filter)


def _circular_mapping(spin):
    c = spin.size / 2.0
    R = spin.size * 0.45

    def to_image(r_frac, theta_deg):
        th = np.radians(np.asarray(theta_deg, dtype=float))
        return (c + R * np.asarray(r_frac) * np.cos(th),
                c + R * np.asarray(r_frac) * np.sin(th))
    return to_image


def _rim_spacetime(spin, frames, times, mapping, t_max):
    pairs = [(float(t), f) for f, t in zip(frames, times) if t <= t_max]
    return build_spacetime(pairs, mapping, 0.86, 1.00, n_bins=720,
                           shape=(spin.size, spin.size)).detrended()


def test_vmf_recovers_ball_velocity_top_down():
    spin = SyntheticSpin(seed=11)
    frames, times, truth = spin.simulate()
    st = _rim_spacetime(spin, frames, times, _circular_mapping(spin),
                        truth.t_drop - 0.2)
    omegas = np.arange(200.0, 2600.0, 25.0)
    res = velocity_matched_filter(st, omegas, np.array([0.0, -200.0]))
    assert res.is_detection(), (res.snr, res.ratio)
    # Mean rim speed over the fitted window, from the generator's own physics.
    assert 400.0 < res.omega < 2400.0, res.omega


def test_vmf_sees_ball_in_low_snr_oblique_view():
    """The decisive case: dim, foreshortened, noisy — blob detection fails."""
    spin = SyntheticSpin(seed=11, elevation_deg=22.0, side_gain=0.40,
                         side_noise=7.0)
    frames, times, truth = spin.simulate()
    st = _rim_spacetime(spin, frames, times, spin.oblique_mapping(),
                        truth.t_drop - 0.2)
    omegas = np.arange(200.0, 2600.0, 25.0)
    ok, res = detector_power(st, omegas, np.array([0.0, -200.0]))
    assert ok, f"blind in oblique view: snr={res.snr:.2f} ratio={res.ratio:.2f}"
    assert 400.0 < res.omega < 2400.0, res.omega


def test_null_rotor_removes_rotor_locked_structure():
    """A synthetic map containing only rotor-locked texture must be nulled."""
    n_bins, n = 720, 120
    t = np.arange(n) / 60.0
    omega = 90.0
    base = np.zeros(n_bins)
    base[::20] = 1.0                       # 36 evenly spaced pockets
    base[0] = 3.0                          # plus a zero marker breaking symmetry
    base = np.convolve(np.r_[base, base, base],
                       np.ones(5) / 5, mode="same")[n_bins:2 * n_bins]
    S = np.array([np.roll(base, int(round(omega * ti / (360.0 / n_bins))))
                  for ti in t])
    st = SpaceTime(S=S, t=t, n_bins=n_bins)
    assert abs(measure_rotor(st) - omega) < 5.0
    nulled = null_rotor(st, omega)
    assert np.abs(nulled.S).max() < 0.05 * np.abs(S - S.mean()).max()


def test_is_detection_rejects_scattered_noise_peaks():
    """Near-equal peaks at unrelated velocities must not count as a find."""
    from prophetvision.vmf import VMFResult
    noisy = VMFResult(snr=4.0, floor=2.0, omega=-800.0, alpha=0.0, phi=10.0,
                      ranking=[(4.0, -800.0, 0.0, 10.0),
                               (3.95, 1900.0, 0.0, 200.0),
                               (3.9, 1500.0, 0.0, 90.0)])
    assert not noisy.is_detection()
    clean = VMFResult(snr=9.0, floor=3.0, omega=-110.0, alpha=0.0, phi=10.0,
                      ranking=[(9.0, -110.0, 0.0, 10.0),
                               (4.0, 1200.0, 0.0, 200.0)])
    assert clean.is_detection()
