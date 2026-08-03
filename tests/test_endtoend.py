import numpy as np
import pytest

from prophetvision.realstream import unwrap_directional, unwrap_predictive


def test_predictive_unwrap_recovers_whole_turns_hidden_in_a_gap():
    """A dropout while the ball turns fast hides entire revolutions.

    Fixed-slack unwrapping silently loses them, which is what made fitted arcs
    jump discontinuously when a few samples were added; the predictive version
    uses the running velocity to pick the right wrap count.
    """
    omega = -900.0                       # deg/s
    t = np.r_[np.arange(0.0, 0.6, 1 / 60), np.arange(1.4, 2.0, 1 / 60)]
    true = omega * t
    obs = true % 360.0

    naive = unwrap_directional(obs, direction=-1)
    good = unwrap_predictive(t, obs, direction=-1)

    true_span = true[-1] - true[0]
    assert abs((good[-1] - good[0]) - true_span) < 15.0
    # the naive version must be off by at least one whole turn
    assert abs((naive[-1] - naive[0]) - true_span) > 300.0


def test_predictive_unwrap_matches_when_there_is_no_gap():
    omega = -320.0
    t = np.arange(0.0, 1.5, 1 / 60)
    true = omega * t
    out = unwrap_predictive(t, true % 360.0, direction=-1)
    assert abs((out[-1] - out[0]) - (true[-1] - true[0])) < 5.0


def test_predictive_unwrap_handles_positive_direction():
    omega = 450.0
    t = np.arange(0.0, 1.2, 1 / 60)
    true = omega * t
    out = unwrap_predictive(t, true % 360.0, direction=+1)
    assert abs((out[-1] - out[0]) - (true[-1] - true[0])) < 6.0


def test_predictive_unwrap_short_input():
    assert len(unwrap_predictive([0.0], [12.0])) == 1


def test_iter_unique_frames_skips_duplicates(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = str(tmp_path / "dup.avi")
    size = 64
    frames = []
    for k in range(12):
        img = np.zeros((size, size, 3), np.uint8)
        img[:, :] = (k // 3) * 20            # changes only every 3rd frame
        frames.append(img)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0,
                         (size, size))
    for f in frames:
        vw.write(f)
    vw.release()

    from prophetvision.endtoend import iter_unique_frames
    got = list(iter_unique_frames(path))
    assert 3 <= len(got) <= 6, f"expected duplicates removed, got {len(got)}"
