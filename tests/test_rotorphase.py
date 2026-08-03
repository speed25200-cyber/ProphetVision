"""The rotor phase must survive the pocket ring's near-periodicity.

Correlating each frame against a fixed reference looks natural but fails here:
the ring repeats every 360/37 degrees, so past one pocket the peak aliases and
the measured phase oscillates around zero instead of ramping (measured on real
footage: a 52 deg/s rotor read as -0.24 deg/s). Consecutive frames differ by
well under one pocket, so integrating frame-to-frame steps is unambiguous.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from prophetvision.rotorphase import RotorPhaseTracker, zero_azimuth


def _wheel_frame(size: int, phi_deg: float, n_pockets: int = 37) -> np.ndarray:
    """Top-down wheel with a green zero pocket at `phi_deg`."""
    img = np.zeros((size, size, 3), np.uint8)
    c = size // 2
    R = int(size * 0.45)
    sector = 360.0 / n_pockets
    for i in range(n_pockets):
        a0 = phi_deg + i * sector - sector / 2
        if i == 0:
            color = (40, 200, 40)          # green zero
        elif i % 2:
            color = (35, 35, 190)
        else:
            color = (25, 25, 25)
        cv2.ellipse(img, (c, c), (int(R * 0.70), int(R * 0.70)), 0,
                    a0, a0 + sector, color, -1)
    cv2.circle(img, (c, c), int(R * 0.50), (60, 60, 65), -1)
    return img


def test_zero_azimuth_finds_the_green_pocket():
    size = 400
    for phi in (0.0, 73.0, 210.0, 315.0):
        img = _wheel_frame(size, phi)
        az, strength = zero_azimuth(img, size / 2, size / 2, size * 0.45)
        err = abs((az - phi + 180) % 360 - 180)
        assert err < 6.0, (phi, az, err)
        assert strength > 2.0


def test_phase_tracks_past_a_full_revolution():
    """More than one turn: a reference-frame correlation would alias here."""
    size, fps, omega = 400, 60.0, 90.0        # 90 deg/s -> 1.5 turns in 6 s
    tr = RotorPhaseTracker(size / 2, size / 2, size * 0.45)
    times = np.arange(0.0, 6.0, 1 / fps)
    for t in times:
        tr.feed(_wheel_frame(size, omega * t), float(t))
    phase = tr.result()
    assert abs(phase.speed_deg_s - omega) < 6.0, phase.speed_deg_s
    assert phase.residual_deg < 12.0, phase.residual_deg


def test_phase_anchor_names_the_zero_pocket():
    size, fps, omega = 400, 60.0, 45.0
    phi0 = 137.0
    tr = RotorPhaseTracker(size / 2, size / 2, size * 0.45)
    for t in np.arange(0.0, 3.0, 1 / fps):
        tr.feed(_wheel_frame(size, phi0 + omega * t), float(t))
    phase = tr.result()
    err = abs((phase.zero_deg[0] - phi0 + 180) % 360 - 180)
    assert err < 8.0, (phase.zero_deg[0], phi0)


def test_stationary_wheel_gives_zero_speed():
    size = 400
    tr = RotorPhaseTracker(size / 2, size / 2, size * 0.45)
    img = _wheel_frame(size, 20.0)
    for k in range(40):
        tr.feed(img, k / 60.0)
    assert abs(tr.result().speed_deg_s) < 3.0
