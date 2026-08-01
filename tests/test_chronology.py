"""Tests for prophetvision/chronology.py (SPEC section D).

Run against the real Gravity Auto Roulette windows in
/mnt/agents/output/prophetvision/videos/. Streaming: frames are read one at
a time and fed to Chronology.process().

Measured facts on these videos (this build):
- hq_060_100.mp4: countdown digits 9->1 at clip t ~= 1.5-10 s (green
  circular timer, right edge), "No more bets" at t ~= 11.6-13.4 s, result
  marker "25" at t ~= 34.3-38.7 s, banner [27,34,29,15,7,...] before the
  spin and [25,27,34,...] after.
- hq_100_135.mp4: "No more bets" at clip t ~= 19.3-21.1 s.
"""
from __future__ import annotations

import os

import cv2
import pytest

from prophetvision.chronology import Chronology, SpinEvent

VIDEOS = "/mnt/agents/output/prophetvision/videos"
CLIP_A = os.path.join(VIDEOS, "hq_060_100.mp4")
CLIP_B = os.path.join(VIDEOS, "hq_100_135.mp4")
CLIP_720 = os.path.join(VIDEOS, "video720.mp4")  # 954x720, same layout

pytestmark = pytest.mark.skipif(
    not (os.path.exists(CLIP_A) and os.path.exists(CLIP_B)),
    reason="real video windows not available",
)

FPS = 60.0


def stream_events(path, t0=0.0, t1=float("inf"), every=1):
    """Feed frames one at a time (streaming) and collect all events."""
    chrono = Chronology()
    cap = cv2.VideoCapture(path)
    events = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / FPS
        if t0 <= t <= t1 and i % every == 0:
            events.extend(chrono.process(t, frame))
        i += 1
    cap.release()
    return chrono, events


def grab(path, t):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * FPS))
    ok, frame = cap.read()
    cap.release()
    assert ok
    return frame


# ----------------------------------------------------------------------
def test_spin_event_dataclass():
    ev = SpinEvent("countdown", 1.5, {"seconds_left": 7})
    assert ev.kind == "countdown" and ev.t == 1.5
    assert ev.detail["seconds_left"] == 7


def test_countdown_ocr_clip_a():
    """Countdown events 9->1 during clip t=0..12 s of hq_060_100.

    The clip starts on the tail of the previous countdown (1, 0), then the
    new spin's countdown runs 9 -> 1. The main decreasing segment must
    contain >= 8 correct consecutive values.
    """
    _, events = stream_events(CLIP_A, t0=0.0, t1=12.0)
    cd = [e for e in events if e.kind == "countdown"]
    values = [e.detail["seconds_left"] for e in cd]
    assert len(cd) >= 8, f"too few countdown events: {values}"
    # longest strictly decreasing-by-one segment
    best, cur = [], [values[0]]
    for a, b in zip(values, values[1:]):
        if b == a - 1:
            cur.append(b)
        else:
            best = max(best, cur, key=len)
            cur = [b]
    best = max(best, cur, key=len)
    assert len(best) >= 8, f"main countdown segment too short: {values}"
    assert best[0] == 9 and best[-1] <= 2, f"unexpected segment: {best}"
    # event times are increasing and values within 0..9
    times = [e.t for e in cd]
    assert times == sorted(times)
    assert all(0 <= v <= 9 for v in values)


def test_bets_close_clip_b():
    """"No more bets" detected between t=15 and t=25 of hq_100_135."""
    _, events = stream_events(CLIP_B, t0=0.0, t1=30.0)
    bc = [e for e in events if e.kind == "bets_close"]
    assert bc, "no bets_close event detected"
    assert any(15.0 <= e.t <= 25.0 for e in bc), \
        f"bets_close at wrong time: {[e.t for e in bc]}"


def test_result_25_clip_a():
    """Result marker OCR reads 25 between t=33 and t=39.5 of hq_060_100."""
    _, events = stream_events(CLIP_A, t0=30.0, t1=40.0)
    res = [e for e in events if e.kind == "result"]
    assert res, "no result event detected"
    good = [e for e in res if e.detail["number"] == 25
            and 33.0 <= e.t <= 39.5]
    assert good, f"result != 25 or wrong time: " \
                 f"{[(e.t, e.detail) for e in res]}"
    # exactly one result event for the spin (anti-duplicate)
    assert len(res) == 1


def test_banner_ocr_clip_a():
    """Banner OCR: >= 12/16 numbers correct on the pre-spin banner."""
    frame = grab(CLIP_A, 2.0)
    numbers = Chronology().read_banner(frame)
    truth = [27, 34, 29, 15, 7, 29, 15, 22, 30, 15, 14, 23, 19, 8, 9, 10]
    correct = sum(1 for a, b in zip(numbers, truth) if a == b)
    assert correct >= 12, f"banner OCR {correct}/16: {numbers}"


def test_banner_event_updates_after_result():
    """A 'banner' event is emitted and shifts to [25, 27, ...] after t=39."""
    _, events = stream_events(CLIP_A, t0=0.0, t1=40.0, every=2)
    banners = [e for e in events if e.kind == "banner"]
    assert banners, "no banner event emitted"
    first = banners[0].detail["numbers"]
    assert first[:5] == [27, 34, 29, 15, 7]
    assert any(b.detail["numbers"][:3] == [25, 27, 34] for b in banners), \
        f"banner never updated to 25: {[b.detail['numbers'][:3] for b in banners]}"


def test_no_spurious_events_on_quiet_segment():
    """No result / bets_close false positives during plain betting view."""
    _, events = stream_events(CLIP_A, t0=14.0, t1=20.0)
    assert not [e for e in events if e.kind in ("result", "bets_close")]


@pytest.mark.skipif(not os.path.exists(CLIP_720),
                    reason="video720.mp4 not available")
def test_resolution_agnostic_720p():
    """Same detector at 954x720 (video720.mp4): scaled ROIs must recover
    the countdown, the results (25 ~= t95, 9 ~= t138-145) and the banner."""
    _, events = stream_events(CLIP_720, t0=88.0, t1=150.0, every=2)
    results = [(e.t, e.detail["number"])
               for e in events if e.kind == "result"]
    assert any(n == 25 and 92.0 <= t <= 100.0 for t, n in results), results
    assert any(n == 9 and 135.0 <= t <= 148.0 for t, n in results), results
    cd = [e for e in events if e.kind == "countdown"]
    assert cd, "no countdown detected at 720p"
    banner = Chronology().read_banner(grab(CLIP_720, 120.0))
    assert banner[:3] == [25, 27, 34], banner
