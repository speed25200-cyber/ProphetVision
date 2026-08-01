"""Tests for prophetvision/live.py (SPEC.md section F + NOTES v2.1).

Real-video session tests run on hq_060_100.mp4 (clip A, 40 s) and
video720.mp4 (954x720, ~4 min) and skip gracefully when the videos are
unavailable.  Measured reality on clip A (NOTES v2.1): plunge at ~21.5 s,
ball on the rim ~22.1-25.5 s, drop ~25.5 s, OCR result 25 at ~35 s.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from prophetvision.config import WheelConfig
from prophetvision.live import (LiveEngine, SessionReport, adaptive_zone,
                                linear_kinematics)

import os
VIDEOS = os.environ.get("PV_VIDEOS", "/mnt/agents/output/prophetvision/videos")
CLIP_A = os.path.join(VIDEOS, "hq_060_100.mp4")
CLIP_720 = os.path.join(VIDEOS, "video720.mp4")

requires_clip_a = pytest.mark.skipif(not os.path.exists(CLIP_A),
                                     reason="real video windows unavailable")
requires_720 = pytest.mark.skipif(not os.path.exists(CLIP_720),
                                  reason="video720.mp4 unavailable")


# ----------------------------------------------------------------------
# unit tests (no video)
# ----------------------------------------------------------------------
def test_linear_kinematics_recovers_ground_truth():
    """Quadratic motion th = w t + .5 a t^2 (w<0, a>0 = decel)."""
    t = np.arange(0.0, 2.0, 0.02)
    th = -120.0 * t + 0.5 * 22.0 * t * t
    kin = linear_kinematics(t, th, window_s=1.5)
    assert kin is not None
    assert kin["omega"] == pytest.approx(-120.0 + 22.0 * t[-1], abs=1.0)
    assert kin["alpha"] == pytest.approx(22.0, abs=1.0)
    # drop estimate: (|w| - 55) / decel from t_now
    dt = (abs(kin["omega"]) - 55.0) / (-np.sign(kin["omega"]) * kin["alpha"])
    assert dt == pytest.approx((120 - 22 * t[-1] - 55) / 22.0, abs=0.05)


def test_linear_kinematics_strength_mask_and_short_input():
    t = np.arange(0.0, 1.0, 0.02)
    th = -100.0 * t
    assert linear_kinematics(t[:4], th[:4]) is None
    # mostly coasted samples (strength 0) -> not enough real support
    s = np.zeros_like(t)
    s[:4] = 10.0
    assert linear_kinematics(t, th, strength=s) is None
    s = np.ones_like(t) * 10.0
    kin = linear_kinematics(t, th, strength=s)
    assert kin is not None and kin["t_last_real"] == pytest.approx(t[-1])


def test_adaptive_zone_width():
    wheel = WheelConfig()
    n = wheel.n_pockets
    delta = np.zeros(n)
    delta[5] = 1.0
    z = adaptive_zone(delta, wheel)
    assert z["width"] == 1 and z["zone_p"] == pytest.approx(1.0)
    uniform = np.full(n, 1.0 / n)
    z = adaptive_zone(uniform, wheel)
    # 0.67 mass of a uniform 37-pocket wheel needs ~25 pockets
    assert 24 <= z["width"] <= 26
    assert z["zone_p"] >= 0.67
    assert all(num in wheel.pocket_order for num in z["zone"])


def test_session_report_summary_and_json(tmp_path):
    rep = SessionReport(video="fake.mp4")
    rep.spins.append({
        "index": 0, "t_plunge_start": 21.5, "t_plunge_end": 39.5,
        "events": [{"kind": "bets_close", "t": 11.6, "detail": {}}],
        "predictions": [{"t_emit": 23.0, "t_drop_est": 25.5, "zone": [1, 2],
                         "zone_p": 0.3, "top_pocket": 1,
                         "impact_pocket_index": 3, "arc_used_s": 1.5,
                         "zone_adaptive": {"width": 5, "zone": [1, 2],
                                           "zone_p": 0.7},
                         "raw_zone": [9]}],
        "drop": {"t": 25.47, "omega_end": 63.8}, "result": 25,
        "hit": True, "hit_adaptive": True, "hit_raw": False,
        "lead_s": 2.5, "track_stats": {"arc_deg": 400.0, "v0": 150.0,
                                       "coverage": 0.95},
        "feasibility": {"launch_minus_bets_close_s": 9.9},
    })
    s = rep.summary()
    assert s["n_spins"] == 1 and s["hits"] == 1 and s["results"] == [25]
    assert s["feasibility"]["betting_feasible"] is False
    out = tmp_path / "r.json"
    rep.save(str(out))
    loaded = json.loads(out.read_text())
    assert loaded["spins"][0]["result"] == 25
    assert loaded["summary"]["n_spins"] == 1


# ----------------------------------------------------------------------
# real session on clip A (fixture in conftest.py, runs once per session)
# ----------------------------------------------------------------------
@requires_clip_a
def test_spin_detected(session_a):
    rep, _ = session_a
    assert len(rep.spins) >= 1
    s = rep.spins[0]
    assert 20.0 <= s["t_plunge_start"] <= 23.0
    assert s["track_stats"]["arc_deg"] >= 250.0
    assert s["track_stats"]["coverage"] >= 0.85


@requires_clip_a
def test_prediction_lead(session_a):
    rep, _ = session_a
    preds = [p for s in rep.spins for p in s["predictions"]]
    assert preds, "no prediction emitted"
    leads = [p["t_drop_est"] - p["t_emit"] for p in preds]
    assert max(leads) >= 2.0, f"best prediction lead {max(leads):.2f} < 2 s"
    # predictions carry the required fields
    p = preds[0]
    for key in ("t_emit", "t_drop_est", "zone", "zone_p", "top_pocket",
                    "impact_pocket_index", "arc_used_s"):
        assert key in p
    assert len(p["zone"]) == 9


@requires_clip_a
def test_drop_and_omega_drop_learning(session_a):
    rep, _ = session_a
    s = rep.spins[0]
    assert s["drop"] is not None
    assert 24.5 <= s["drop"]["t"] <= 26.5  # measured 25.47
    # omega_drop learned online from the track end (NOTES v2.1: ~50-65)
    assert 40.0 <= rep.omega_drop_final <= 90.0


@requires_clip_a
def test_result_25_associated(session_a):
    rep, _ = session_a
    assert any(s["result"] == 25 for s in rep.spins)
    s = [s for s in rep.spins if s["result"] == 25][0]
    assert s["hit"] is not None
    assert "offset_raw" in s and "hit_raw" in s and "hit_adaptive" in s


@requires_clip_a
def test_feasibility_gap_positive(session_a):
    """Honesty: the ball is launched AFTER bets close on this stream."""
    rep, _ = session_a
    s = rep.spins[0]
    gap = s["feasibility"]["launch_minus_bets_close_s"]
    assert gap is not None and gap > 0
    assert rep.summary()["feasibility"]["betting_feasible"] is False


@requires_clip_a
def test_json_written(session_a, tmp_path):
    rep, _ = session_a
    out = tmp_path / "session.json"
    rep.save(str(out))
    loaded = json.loads(out.read_text())
    assert loaded["summary"]["results"] == [25]
    assert loaded["spins"][0]["predictions"]


@requires_clip_a
def test_ram_below_1gb(session_a):
    _, ram_mb = session_a
    assert ram_mb < 1024.0, f"peak RAM delta {ram_mb:.0f} MB"


@requires_clip_a
def test_pipeline_perf(session_a):
    rep, _ = session_a
    assert rep.pipeline_fps >= 30.0, f"{rep.pipeline_fps:.1f} fps < 30"


@requires_clip_a
def test_realtime_mode_not_faster_than_real_time():
    eng = LiveEngine(realtime=True, max_seconds=2.0)
    t0 = time.perf_counter()
    eng.run(CLIP_A)
    assert time.perf_counter() - t0 >= 1.8


@requires_clip_a
def test_on_event_callback_receives_predictions():
    events = []
    LiveEngine(max_seconds=28.0).run(CLIP_A, on_event=events.append)
    kinds = {e["kind"] for e in events}
    assert "prediction" in kinds and "launch" in kinds


# ----------------------------------------------------------------------
# resolution robustness: full 720p session (SPEC F perf + lead finding #3)
# ----------------------------------------------------------------------
@requires_720
def test_full_720p_session_no_crash_and_spins():
    rep = LiveEngine().run(CLIP_720)
    assert len(rep.spins) >= 3, f"only {len(rep.spins)} spins on video720"
    assert rep.pipeline_fps >= 30.0
    # at least one spin produced a ball track + prediction
    assert any(s["predictions"] for s in rep.spins)
