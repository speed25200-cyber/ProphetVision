import numpy as np

from prophetvision.geometry import (circular_shift_deg, sample_ring,
                                    unwrap_deg, wrap_deg)


def test_circular_shift_recovers_known_rotation():
    n = 720
    x = np.arange(n)
    base = np.sin(2 * np.pi * x / n * 3) + 0.5 * np.sin(2 * np.pi * x / n * 7)
    for shift_bins in (5, -8, 13):
        rotated = np.roll(base, shift_bins)
        est = circular_shift_deg(base, rotated)
        assert abs(est - shift_bins * 0.5) < 0.1


def test_circular_shift_window_suppresses_alias():
    # Strongly periodic pattern (period n/8) with a small true shift: the
    # windowed search must not jump to an aliased peak.
    n = 720
    x = np.arange(n)
    base = np.sin(2 * np.pi * x / n * 8)
    rotated = np.roll(base, 4)
    est = circular_shift_deg(base, rotated, max_shift_deg=10.0)
    assert abs(est - 2.0) < 0.2


def test_sample_ring_sees_bright_sector():
    img = np.zeros((200, 200), np.float32)
    # Bright wedge around azimuth 90 deg (down in image coords).
    yy, xx = np.mgrid[0:200, 0:200]
    ang = np.degrees(np.arctan2(yy - 100.0, xx - 100.0)) % 360
    rr = np.hypot(xx - 100.0, yy - 100.0)
    img[(np.abs(ang - 90) < 10) & (rr > 55) & (rr < 65)] = 255
    prof = sample_ring(img, 100, 100, 60, 8, n_bins=360)
    bright = np.nonzero(prof > 0.9 * prof.max())[0]
    assert abs(float(bright.mean()) - 90) < 4


def test_unwrap():
    a = wrap_deg(np.arange(0, 2000, 37.0))
    u = unwrap_deg(a)
    assert np.allclose(np.diff(u), 37.0, atol=1e-6)
