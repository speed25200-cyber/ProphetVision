import numpy as np
import pytest

from prophetvision.config import EUROPEAN_ORDER, WheelConfig
from prophetvision.pocketmap import PocketMap, calibrate_offset


def test_offset_recovers_a_known_shift():
    wheel = WheelConfig()
    true_offset = 7
    for pocket in (0, 25, 9, 17):
        measured = (wheel.index_of(pocket) - true_offset) % wheel.n_pockets
        assert calibrate_offset(measured, pocket, wheel) == pytest.approx(
            true_offset, abs=1e-6)


def test_map_names_pockets_after_calibration():
    pm = PocketMap()
    wheel = pm.wheel
    true_offset = 11
    for pocket in (25, 9, 31):
        pm.observe((wheel.index_of(pocket) - true_offset) % wheel.n_pockets,
                   pocket)
    assert pm.is_trustworthy, pm.spread_pockets
    for pocket in (0, 14, 36):
        measured = (wheel.index_of(pocket) - true_offset) % wheel.n_pockets
        assert pm.pocket(measured) == pocket


def test_uncalibrated_map_refuses_to_name():
    pm = PocketMap()
    with pytest.raises(RuntimeError):
        pm.pocket(3.0)


def test_inconsistent_labels_are_flagged_untrustworthy():
    """Two spins disagreeing by half the wheel -- exactly what the real
    settled-ball measurement produced -- must not pass as calibrated."""
    pm = PocketMap()
    pm.observe(36.4, 25)     # offset ~8
    pm.observe(0.0, 9)       # offset 27
    assert pm.offset is not None
    assert pm.spread_pockets > 1.0
    assert not pm.is_trustworthy


def test_zone_is_centred_and_the_right_width():
    pm = PocketMap()
    pm.observe(0.0, 0)
    pm.observe(1.0, 32)      # consistent: offset 0 for both
    z = pm.zone(0.0, width=9)
    assert len(z) == 9
    assert z[4] == 0                      # centred on the named pocket
    assert z == [EUROPEAN_ORDER[(i) % 37] for i in range(-4, 5)]


def test_wraparound_offsets_average_circularly():
    """Offsets near the 0/37 seam must not average to the middle of the wheel."""
    pm = PocketMap()
    wheel = pm.wheel
    for pocket, off in ((25, 36.5), (9, 0.5)):
        pm.observe((wheel.index_of(pocket) - off) % wheel.n_pockets, pocket)
    assert pm.spread_pockets < 1.5
    assert min(pm.offset, 37 - pm.offset) < 1.5
