"""Tests for prophetvision.streaming (SPEC A) on the real test videos."""
from __future__ import annotations

import os
import resource

import numpy as np
import pytest

from prophetvision.streaming import FrameSource, ShotSegmenter, dedup_times

VIDEOS = "/mnt/agents/output/prophetvision/videos"
CLIP_A = os.path.join(VIDEOS, "hq_060_100.mp4")   # oblique 0-21.5 s, plunge 21.5-39.5 s
CLIP_B = os.path.join(VIDEOS, "hq_100_135.mp4")   # oblique, "No more bets" ~20 s

pytestmark = pytest.mark.skipif(
    not os.path.isfile(CLIP_A), reason="test videos not available"
)


# ---------------------------------------------------------------- FrameSource
def test_frame_source_properties_and_iteration():
    src = FrameSource(CLIP_A)
    assert src.fps == pytest.approx(60.0, abs=1.0)
    count = 0
    prev_idx, prev_t = -1, -1.0
    for idx, t, frame in src:
        assert idx == prev_idx + 1
        assert t > prev_t
        assert t == pytest.approx(idx / src.fps)
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.shape[0] > 500 and frame.shape[1] > 500
        prev_idx, prev_t = idx, t
        count += 1
        if count >= 90:
            break
    assert count == 90


def test_frame_source_reiterates_from_start():
    from itertools import islice

    src = FrameSource(CLIP_A)
    it1 = [(idx, t) for idx, t, _f in islice(src, 3)]
    it2 = [(idx, t) for idx, t, _f in islice(src, 3)]
    assert [i for i, _t in it1] == [0, 1, 2]
    assert it1 == it2
    assert it1[1][1] == pytest.approx(1 / 60.0)


def test_frame_source_independent_iterators():
    """Two iterators over the same FrameSource must not interleave reads."""
    from itertools import islice

    src = FrameSource(CLIP_A)
    it_a, it_b = iter(src), iter(src)
    ia, ta, fa = next(it_a)
    ib, tb, fb = next(it_b)
    assert ia == ib == 0 and ta == tb == 0.0
    assert np.array_equal(fa, fb)
    ja, _ta, fa2 = next(it_a)
    assert ja == 1
    # iterator B is unaffected by A's reads: it still yields frame 1 next
    jb, _tb, fb2 = next(it_b)
    assert jb == 1 and np.array_equal(fa2, fb2)


def test_frame_source_streaming_ram():
    """Iterating the whole 40 s clip must not accumulate frames in RAM."""
    src = FrameSource(CLIP_A)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    n = 0
    acc = 0.0
    for _idx, _t, frame in src:
        acc += float(frame[0, 0, 0])  # touch the frame, then drop it
        n += 1
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert n > 2000
    assert acc > 0
    # Peak RSS increase well below the 500 Mo budget (a frame is ~3.3 Mo).
    assert (rss_after - rss_before) < 500 * 1024  # ru_maxrss is in KiB


def test_frame_source_missing_file():
    with pytest.raises((FileNotFoundError, IOError)):
        FrameSource(os.path.join(VIDEOS, "does_not_exist.mp4"))


# ----------------------------------------------------------------- dedup
def test_dedup_times_synthetic():
    f0 = np.zeros((226, 300, 3), np.uint8)
    f1 = np.full((226, 300, 3), 200, np.uint8)
    frames = [(i, i / 60.0, f0 if i % 2 == 0 else f1) for i in range(10)]
    # alternating frames are all "new" (large diff)
    out = list(dedup_times(iter(frames)))
    assert len(out) == 10
    # exact duplicates: only the first of each run survives
    frames2 = [(i, i / 60.0, f0.copy()) for i in range(10)]
    out2 = list(dedup_times(iter(frames2)))
    assert len(out2) == 1 and out2[0][0] == 0.0


def test_dedup_times_real_video():
    """The 60 fps container holds duplicated images; dedup must drop them and
    keep correct (first-occurrence) times."""
    src = FrameSource(CLIP_A)
    ts = []
    for t, frame in dedup_times(iter(src)):
        ts.append(t)
        del frame  # never hold frames: streaming only
    n_total = src.n_frames
    n_uniq = len(ts)
    # Measured: ~2000 unique frames out of 2401 (~50 real fps).
    assert 0.70 * n_total < n_uniq < 0.95 * n_total
    assert ts == sorted(ts) and len(set(ts)) == len(ts)
    # Times stay aligned with the container clock (first occurrence).
    assert ts[0] == pytest.approx(0.0)
    assert ts[-1] == pytest.approx(src.duration, abs=0.05)


# ------------------------------------------------------------ ShotSegmenter
@pytest.fixture(scope="module")
def clip_a_states():
    src = FrameSource(CLIP_A)
    seg = ShotSegmenter()
    return [(t, seg.process(frame)) for _idx, t, frame in src]


def test_segmentation_clip_a_cut_at_21_5(clip_a_states):
    states = clip_a_states
    # first sustained plunge state
    plunge_ts = [t for t, s in states if s == "plunge"]
    assert plunge_ts, "no plunge state detected"
    first_plunge = plunge_ts[0]
    assert first_plunge == pytest.approx(21.5, abs=1.0)
    # before the cut: essentially everything oblique
    before = [s for t, s in states if t < first_plunge - 0.5]
    assert before.count("oblique") / len(before) >= 0.95
    # after the cut (and before the clip end): essentially everything plunge
    after = [s for t, s in states if first_plunge + 1.0 < t < 39.0]
    assert after.count("plunge") / len(after) >= 0.95


def test_segmentation_no_flicker(clip_a_states):
    """Hysteresis: at most 2 distinct state runs on clip A (oblique->plunge)."""
    runs = 1
    prev = clip_a_states[0][1]
    for _t, s in clip_a_states[1:]:
        if s != prev:
            runs += 1
            prev = s
    assert runs <= 2


@pytest.mark.skipif(not os.path.isfile(CLIP_B), reason="clip B not available")
def test_segmentation_clip_b_no_more_bets_overlay():
    """The 'No more bets' text overlay (~t=20 s) must not flip the state."""
    src = FrameSource(CLIP_B)
    seg = ShotSegmenter()
    states = [(t, seg.process(frame)) for _idx, t, frame in src]
    mid = [s for t, s in states if 17.0 < t < 23.0]
    assert set(mid) == {"oblique"}
