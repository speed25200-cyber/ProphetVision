"""Tests for prophetvision.balltrack (SPEC.md section C).

Measurable requirement (corrected, see note): on plunge_A.mp4 the plunge
shot only shows the END of the spin — the ball runs the rim annulus from
t~0.3 s to its drop at t~4.0-4.2 s, decelerating ~125 -> ~55 deg/s
(verified against frame-difference ground truth and against the reference
proof-of-concept explore/track_median.py, whose own stdout shows the same
~100 deg/s slope).  The SPEC's "800 deg/s, arc >= 4000 deg over 13 s"
figures came from a misreading of 6-frame-strided samples as per-frame
steps (factor 6) and are physically impossible on this clip; the
thresholds below are the measured reality (approved by the lead):
coverage >= 90 % of the ball window, arc >= 300 deg, |v0| in [70, 180]
deg/s, monotonicity >= 98 %, < 2 % abnormal jumps, clean drop (no
phantom track afterwards).

Sub-bin precision (<= 0.5 deg) is asserted on a synthetic sequence.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from prophetvision.balltrack import BallSample, BallTracker, PolarMedianTracker
from prophetvision.calibration import ObliqueCal, PlungeCal

VIDEO = "/mnt/agents/output/prophetvision/videos/plunge_A.mp4"
CAL = PlungeCal(cx=610.0, cy=419.0, R=345.0)
FPS = 60.0
ARTIFACTS = Path(__file__).parent / "artifacts"

JITTER_DEG = 0.5          # sub-bin precision: monotonicity deadband
BALL_WIN = (0.3, 4.05)    # measured window with the ball on the rim


# ----------------------------------------------------------------------
# synthetic data helpers
# ----------------------------------------------------------------------
def _draw_blob(img, x, y, value=255, radius=5):
    cv2.circle(img, (int(round(x)), int(round(y))), radius,
               int(value), -1, lineType=cv2.LINE_AA)


def synth_frame(shape, cal, ball_az=None, ball_prev_az=None, r_frac=0.93,
                spots=(), noise_rng=None, noise=3.0):
    """Gray frame with a (motion-blurred) ball on the rim circle and
    optional static spots.  ball_az in degrees (image convention, y down).
    """
    img = np.full(shape, 32, np.uint8)
    if noise_rng is not None:
        img = img + noise_rng.integers(0, int(noise) + 1, shape,
                                       dtype=np.uint8)
    for sx, sy, val in spots:
        _draw_blob(img, sx, sy, value=val, radius=4)
    if ball_az is not None:
        th = np.radians(ball_az)
        x = cal.cx + cal.R * r_frac * np.cos(th)
        y = cal.cy + cal.R * r_frac * np.sin(th)
        if ball_prev_az is not None:      # motion-blur streak
            th0 = np.radians(ball_prev_az)
            x0 = cal.cx + cal.R * r_frac * np.cos(th0)
            y0 = cal.cy + cal.R * r_frac * np.sin(th0)
            cv2.line(img, (int(round(x0)), int(round(y0))),
                     (int(round(x)), int(round(y))), 220, 6,
                     lineType=cv2.LINE_AA)
        _draw_blob(img, x, y, value=255, radius=5)
    return img


def run_tracker(trk, frames, fps=FPS, t0=0.0):
    for i, fr in enumerate(frames):
        trk.process(t0 + i / fps, fr)
    return trk


def make_ball_frames(n, fps, omega0, alpha, shape=(910, 1206), cal=CAL,
                     az0=200.0, blackout=(), spots=(), seed=0):
    """Frames of a ball at az(t) = az0 + omega0 t + alpha t^2 / 2 (deg)."""
    rng = np.random.default_rng(seed)
    frames = []
    prev_az = None
    for i in range(n):
        t = i / fps
        az = az0 + omega0 * t + 0.5 * alpha * t * t
        draw = not any(a <= i < b for a, b in blackout)
        fr = synth_frame(shape, cal,
                         ball_az=az if draw else None,
                         ball_prev_az=prev_az if draw else None,
                         spots=spots, noise_rng=rng)
        frames.append(fr)
        prev_az = az if draw else None
    return frames


# ----------------------------------------------------------------------
# synthetic tests
# ----------------------------------------------------------------------
class TestSynthetic:
    def test_follows_moving_ball_subbin(self):
        """Track a decelerating ball; recovered track must match truth to
        sub-bin precision and be strictly monotone."""
        fps, n = 60.0, 240
        omega0, alpha, az0 = -150.0, 20.0, 200.0
        frames = make_ball_frames(n, fps, omega0, alpha, az0=az0)
        trk = run_tracker(PolarMedianTracker(CAL, fps), frames, fps)
        segs = trk.segments
        assert len(segs) == 1
        st, sth = segs[0]
        assert len(st) >= 0.9 * n
        # compare against truth (skip the first 0.2 s of filter convergence)
        m = st > st[0] + 0.2
        truth = az0 + omega0 * st[m] + 0.5 * alpha * st[m] ** 2
        err = sth[m] - truth
        assert np.sqrt(np.mean(err**2)) < 0.5          # sub-bin precision
        # monotonic decreasing (azimut decroissant)
        assert (np.diff(sth) <= JITTER_DEG).mean() == 1.0
        # recovered initial speed in the right range (fit over the 1st s)
        m1 = st <= st[0] + 1.0
        v = np.polyfit(st[m1], sth[m1], 1)[0]
        assert -170 < v < -110

    def test_static_blob_rejected(self):
        """A pulsing static spot (not absorbed by the median) must NOT
        produce a validated segment, and must be init-blacklisted."""
        fps, n = 60.0, 180
        rng = np.random.default_rng(1)
        th = np.radians(41.0)           # the real video's trap azimuth
        sx = CAL.cx + CAL.R * 0.93 * np.cos(th)
        sy = CAL.cy + CAL.R * 0.93 * np.sin(th)
        frames = []
        for i in range(n):
            spots = [(sx, sy, 230)] if i % 2 == 0 else []
            frames.append(synth_frame((910, 1206), CAL, spots=spots,
                                      noise_rng=rng))
        trk = run_tracker(PolarMedianTracker(CAL, fps), frames, fps)
        assert trk.segments == []
        assert trk.drop_time is None

    def test_dropout_bridged_and_long_gap_cut(self):
        """A 0.3 s blackout must be bridged (one segment); a 1.0 s blackout
        must cut the track into two segments."""
        fps, n = 60.0, 300
        frames = make_ball_frames(n, fps, -140.0, 15.0, az0=250.0,
                                  blackout=[(90, 108)])     # 0.3 s
        trk = run_tracker(PolarMedianTracker(CAL, fps), frames, fps)
        assert len(trk.segments) == 1
        st, _ = trk.segments[0]
        assert st[0] < 90 / fps and st[-1] > 108 / fps      # bridged

        frames = make_ball_frames(n, fps, -140.0, 15.0, az0=250.0,
                                  blackout=[(90, 150)])     # 1.0 s
        trk = run_tracker(PolarMedianTracker(CAL, fps), frames, fps)
        assert len(trk.segments) == 2

    def test_jump_rejection(self):
        """A bright distractor blob > 45 deg away must not steal the track."""
        fps, n = 60.0, 180
        rng = np.random.default_rng(2)
        base = make_ball_frames(n, fps, -130.0, 10.0, az0=210.0, seed=3)
        for i in range(60, 120):                            # distractor
            t = i / fps
            az = np.radians((210.0 - 130.0 * t + 5 * t * t + 120.0) % 360)
            x = CAL.cx + CAL.R * 0.93 * np.cos(az)
            y = CAL.cy + CAL.R * 0.93 * np.sin(az)
            _draw_blob(base[i], x, y, value=240, radius=4)
        trk = run_tracker(PolarMedianTracker(CAL, fps), base, fps)
        assert len(trk.segments) == 1
        st, sth = trk.segments[0]
        steps = np.abs(np.diff(sth))
        assert steps.max() < 45.0
        # still monotone decreasing
        assert (np.diff(sth) <= JITTER_DEG).mean() >= 0.99

    def test_45deg_jump_rule(self):
        """Hard rule: a measurement more than 45 deg away from the
        prediction is never accepted."""
        fps, n = 60.0, 120
        frames = make_ball_frames(n, fps, -130.0, 10.0, az0=210.0, seed=4)
        trk = PolarMedianTracker(CAL, fps)
        run_tracker(trk, frames, fps)
        # feed an outlier 100 deg away while tracking
        trk2 = PolarMedianTracker(CAL, fps)
        run_tracker(trk2, frames[:60], fps)
        assert trk2._x is not None
        pred = float(trk2._x[0])
        outlier = synth_frame((910, 1206), CAL, ball_az=(pred + 100) % 360,
                              noise_rng=np.random.default_rng(5))
        trk2.process(61 / fps, outlier)
        # state must not have jumped to the outlier
        assert abs((trk2._x[0] - pred + 180) % 360 - 180) < 10.0


# ----------------------------------------------------------------------
# real video: the measurable requirement of SPEC C (corrected thresholds)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def tracked():
    if not Path(VIDEO).exists():
        pytest.skip("plunge_A.mp4 missing")
    trk = PolarMedianTracker(CAL, FPS)
    cap = cv2.VideoCapture(VIDEO)
    i = 0
    t0 = time.perf_counter()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        trk.process(i / FPS, fr)
        i += 1
    cap.release()
    proc_s = time.perf_counter() - t0
    t_all, th_all = trk.track()
    # artifact for the live/dashboard agents
    ARTIFACTS.mkdir(exist_ok=True)
    np.savez(ARTIFACTS / "ball_track_plunge_A.npz",
             t=t_all, theta_unwrapped_deg=th_all,
             drop_time=np.array(trk.drop_time),
             segment_start=np.array([s[0][0] for s in trk.segments]),
             segment_end=np.array([s[0][-1] for s in trk.segments]))
    return trk, i, proc_s

def test_segments_cover_ball_window(tracked):
    trk, n_frames, _ = tracked
    segs = trk.segments
    assert segs, "no segment produced"
    present = np.zeros(n_frames, bool)
    for st, _sth in segs:
        idx = np.clip(np.round(st * FPS).astype(int), 0, n_frames - 1)
        present[idx] = True
    win = np.arange(int(BALL_WIN[0] * FPS), int(BALL_WIN[1] * FPS))
    assert present[win].mean() >= 0.90


def test_arc_and_initial_speed(tracked):
    trk, _, _ = tracked
    segs = trk.segments
    arc = sum(abs(sth[-1] - sth[0]) for _st, sth in segs)
    assert arc >= 300.0
    st, sth = segs[0]
    m = st <= st[0] + 0.5
    v0 = np.polyfit(st[m], sth[m], 1)[0]
    assert -180.0 <= v0 <= -70.0        # azimut decroissant


def test_monotonicity_and_jumps(tracked):
    trk, _, _ = tracked
    back = tot = ab = 0
    for _st, sth in trk.segments:
        d = np.diff(sth)
        back += int((d > JITTER_DEG).sum())
        tot += len(d)
        da = np.abs(d)
        med = np.median(da[da > 0]) if (da > 0).any() else 1.0
        ab += int((da > 3 * med).sum())
    assert 1 - back / tot >= 0.98
    assert ab / tot < 0.02


def test_clean_drop_no_phantom(tracked):
    trk, _, _ = tracked
    drop = trk.drop_time
    assert drop is not None
    assert 3.8 <= drop <= 4.3            # measured ball drop
    for st, _sth in trk.segments:
        assert st[0] < 5.5, "phantom segment after the drop"


def test_performance(tracked):
    _trk, n_frames, proc_s = tracked
    assert n_frames / proc_s >= 30.0


def test_track_format(tracked):
    trk, _, _ = tracked
    t_all, th_all = trk.track()
    assert isinstance(t_all, np.ndarray) and isinstance(th_all, np.ndarray)
    assert t_all.shape == th_all.shape
    # NaN separator between the two segments
    assert np.isnan(t_all).sum() == max(0, len(trk.segments) - 1)


# ----------------------------------------------------------------------
# BallTracker facade
# ----------------------------------------------------------------------
class TestFacade:
    def test_plunge_and_pause_and_recalib(self):
        fps = 60.0
        bt = BallTracker(fps)
        frames = make_ball_frames(120, fps, -140.0, 15.0, az0=250.0)
        bt.process(0.0, frames[0])           # no shot set: ignored
        bt.set_shot("plunge", CAL)
        for i, fr in enumerate(frames):
            bt.process(i / fps, fr)
        bt.set_shot("other")                 # pause tracking
        bt.process(2.5, frames[0])
        segs = bt.segments
        assert len(segs) == 1
        st, sth = segs[0]
        assert abs(sth[-1] - sth[0]) > 100
        # recalibration on a new plunge shot starts a fresh tracker
        bt.set_shot("plunge", CAL)
        frames2 = make_ball_frames(120, fps, -140.0, 15.0, az0=80.0)
        for i, fr in enumerate(frames2):
            bt.process(3.0 + i / fps, fr)
        assert len(bt.segments) == 2
        t_all, th_all = bt.track()
        assert t_all.shape == th_all.shape
        assert np.isnan(t_all).sum() == 1    # one NaN separator
        assert bt.drop_time is not None

    def test_oblique_best_effort(self):
        """ObliqueCal ellipse mapping: a ball running the ellipse must be
        tracked in canonical azimuth."""
        fps = 60.0
        cal = ObliqueCal(cx=600.0, cy=430.0, a=440.0, b=180.0, angle=8.0)
        rng = np.random.default_rng(6)
        frames = []
        for i in range(150):
            t = i / fps
            th = np.radians(200.0 - 120.0 * t)
            ca, sa = np.cos(np.radians(cal.angle)), np.sin(np.radians(cal.angle))
            x0, y0 = cal.a * 0.93 * np.cos(th), cal.b * 0.93 * np.sin(th)
            x = cal.cx + ca * x0 - sa * y0
            y = cal.cy + sa * x0 + ca * y0
            img = np.full((910, 1206), 32, np.uint8)
            img = img + rng.integers(0, 4, img.shape, dtype=np.uint8)
            _draw_blob(img, x, y, value=255, radius=5)
            frames.append(img)
        bt = BallTracker(fps)
        bt.set_shot("oblique", cal)
        for i, fr in enumerate(frames):
            bt.process(i / fps, fr)
        segs = bt.segments
        assert len(segs) == 1
        st, sth = segs[0]
        assert sth[-1] < sth[0]              # decreasing canonical azimuth
        assert abs(sth[-1] - sth[0]) > 150   # most of the 300 deg arc
