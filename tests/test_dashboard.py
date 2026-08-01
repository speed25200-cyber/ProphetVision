"""Tests for prophetvision/dashboard.py (SPEC.md section G).

Integration tests reuse the shared clip-A session (conftest fixture) and
skip when the video is unavailable; unit tests build fabricated reports.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from prophetvision.config import WheelConfig
from prophetvision.dashboard import (omega_svg, render_spin_report,
                                     wheel_svg)
from prophetvision.live import SessionReport

VIDEOS = os.environ.get("PV_VIDEOS", "/mnt/agents/output/prophetvision/videos")
CLIP_A = os.path.join(VIDEOS, "hq_060_100.mp4")

requires_clip_a = pytest.mark.skipif(not os.path.exists(CLIP_A),
                                     reason="real video windows unavailable")


# ----------------------------------------------------------------------
# unit tests
# ----------------------------------------------------------------------
def test_wheel_svg_marks_zone_top_and_result():
    wheel = WheelConfig()
    probs = np.full(wheel.n_pockets, 1 / wheel.n_pockets)
    svg = wheel_svg(wheel, list(probs), zone=[0, 32, 15, 19, 4, 21, 2, 25,
                                              17],
                    zone_adaptive={"width": 13,
                                   "zone": list(range(13)), "zone_p": 0.68},
                    top_pocket=21, result=25)
    assert svg.count("<path") > wheel.n_pockets  # pockets + arcs + marker
    assert "circle" in svg  # result marker


def test_omega_svg_empty_series():
    assert "pas de piste" in omega_svg({"series": None})


def _fabricated_report():
    wheel = WheelConfig()
    n = wheel.n_pockets
    probs = [round(1.0 / n, 5)] * n
    rep = SessionReport(video="fake.mp4", n_frames=100, wall_s=2.0)
    rep.scatter = {"n_observations": 1.0, "sigma_pockets": 2.0,
                   "counts": [1.0] * n, "distribution": probs}
    rep.spins.append({
        "index": 0, "t_plunge_start": 10.0, "t_plunge_end": 20.0,
        "calibration": {"cx": 610.0, "cy": 419.0, "R": 345.0},
        "events": [
            {"kind": "countdown", "t": 6.8, "detail": {"seconds_left": 1}},
            {"kind": "bets_close", "t": 11.6, "detail": {"energy": 2000}},
            {"kind": "launch", "t": 10.0, "detail": {"shot": "plunge"}},
        ],
        "predictions": [{
            "t_emit": 12.0, "t_drop_est": 14.5, "zone": [0, 32, 15, 19, 4,
                                                         21, 2, 25, 17],
            "zone_p": 0.24, "top_pocket": 0, "impact_pocket_index": 5,
            "arc_used_s": 1.4, "omega_now": 95.0, "theta_impact_deg": 42.0,
            "probs": probs, "raw_zone": [8, 23, 10, 5, 24, 16, 33, 1, 20],
            "zone_adaptive": {"width": 13, "zone": list(range(13)),
                              "zone_p": 0.68},
            "lead_actual_s": 2.3,
        }],
        "drop": {"t": 14.3, "omega_end": 60.0, "arc_deg": 350.0},
        "result": 25, "hit": True, "hit_adaptive": True, "hit_raw": False,
        "impact_pocket_index": 5, "offset_raw": 2, "lead_s": 2.5,
        "track_stats": {"arc_deg": 350.0, "v0": 140.0, "coverage": 0.95},
        "feasibility": {"launch_minus_bets_close_s": 8.4},
        "series": {"t_ball": [11.0, 12.0, 13.0, 14.0],
                   "theta_ball": [0.0, -100.0, -190.0, -270.0],
                   "omega_ball": [-110.0, -95.0, -82.0, -70.0],
                   "t_rotor": [11.0, 12.0, 13.0, 14.0],
                   "phi_rotor": [0.0, 58.0, 116.0, 174.0],
                   "omega_rotor": [58.0, 58.0, 58.0, 58.0]},
    })
    return rep


def test_render_fabricated_report(tmp_path):
    rep = _fabricated_report()
    out = tmp_path / "r.html"
    # video path is only used for fps + frame grabs; missing file must not
    # crash the renderer (frames are then skipped)
    render_spin_report(rep, str(tmp_path / "no_such_video.mp4"), str(out),
                       str(tmp_path / "frames"))
    txt = out.read_text()
    assert "roue" in txt and "chronologie" in txt
    assert "faisabilité" in txt.lower() or "faisabilit" in txt


# ----------------------------------------------------------------------
# integration on the real clip-A session
# ----------------------------------------------------------------------
@requires_clip_a
def test_report_file_size_and_content(session_a, tmp_path):
    rep, _ = session_a
    out = tmp_path / "report.html"
    frames = tmp_path / "frames"
    render_spin_report(rep, CLIP_A, str(out), str(frames))
    txt = out.read_text()
    assert out.stat().st_size > 30_000, f"{out.stat().st_size} bytes"
    spin = rep.spins[0]
    zone = spin["predictions"][-1]["zone"]
    # the zone numbers appear in the report
    for num in zone:
        assert str(num) in txt
    assert "25" in txt                      # OCR result
    assert "faisabilit" in txt              # honest verdict banner
    assert "APRÈS la fermeture des paris" in txt
    assert "<svg" in txt and "polyline" in txt  # wheel + omega curves
    assert "chute" in txt


@requires_clip_a
def test_annotated_frames_written(session_a, tmp_path):
    rep, _ = session_a
    out = tmp_path / "report.html"
    frames = tmp_path / "frames"
    render_spin_report(rep, CLIP_A, str(out), str(frames))
    jpgs = list(frames.glob("*.jpg"))
    assert jpgs, "no annotated frame written"
    assert all(j.stat().st_size > 10_000 for j in jpgs)
    txt = out.read_text()  # frames referenced relatively from the HTML
    for j in jpgs:
        assert j.name in txt
