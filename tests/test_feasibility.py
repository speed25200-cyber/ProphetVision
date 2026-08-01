import os
import tempfile

from prophetvision.feasibility import audit
from prophetvision.synthetic import SyntheticSpin, write_video


def test_audit_passes_on_clean_synthetic_spin():
    spin = SyntheticSpin(seed=3)
    frames, _, truth = spin.simulate()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "spin.avi")
        write_video(path, frames, spin.fps)
        rep = audit(path, end_s=truth.t_drop - 0.5)
    # A long, well-sampled rim phase must clear every precondition.
    assert rep.unique_samples >= 30, rep.summary()
    assert rep.ball_arc_deg >= 720, rep.summary()
    assert rep.predictable, rep.summary()


def test_audit_rejects_a_too_short_window():
    """Half a second of rim phase cannot identify the decay model."""
    spin = SyntheticSpin(seed=3)
    frames, _, truth = spin.simulate()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "spin.avi")
        write_video(path, frames, spin.fps)
        rep = audit(path, start_s=truth.t_drop - 0.9, end_s=truth.t_drop - 0.4)
    assert not rep.predictable
    assert rep.reasons
