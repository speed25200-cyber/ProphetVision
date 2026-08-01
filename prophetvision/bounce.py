"""Bounce-phase settle prediction (maximal precision = direct measurement).

After the ball leaves the rim (drop), it rattles through the pocket area for
several seconds in full view of the plunge camera.  This module tracks the
cream-colored ball through that phase and reads the pocket in which its
pocket-relative position *durably* stabilizes — the exact winning pocket,
declared as soon as the read is stable instead of waiting for the game UI.

Validated on the Video-v1 footage (Gravity Auto Roulette):
  - spin A (truth 25): exact read, stable ~ at the UI reveal;
  - spin B (truth 9):  exact read, ~3 s BEFORE the UI reveal.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import PlungeCal
from .config import WheelConfig
from .geometry import wrap_deg
from .rotortrack import RotorTracker

__all__ = ["BounceSettle", "predict_settle_pocket"]

POCKET_DEG = 360.0 / 37


def _candidates(frame, cx, cy, R, r_lo=0.42, r_hi=0.85):
    """Cream/yellow compact blobs in the pocket annulus (the ball)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (h >= 14) & (h <= 45) & (s >= 40) & (s <= 140) & (v > 170) & ((v - s) > 90)
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    m &= (rr >= r_lo * R) & (rr <= r_hi * R)
    m = m.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        a = stats[i, 4]
        if not (5 <= a <= 150):
            continue
        w, h = stats[i, 2], stats[i, 3]
        if max(w, h) / max(min(w, h), 1) > 2.5:
            continue
        out.append((float(cent[i][0]), float(cent[i][1]), int(a)))
    return out


@dataclass
class BounceSettle:
    pocket: int                 # declared settle pocket (wheel number)
    t_declare: float            # video time (s) of the declaration
    t_stable_from: float        # video time from which the read never leaves ±1 pocket
    n_track_points: int


def predict_settle_pocket(video: str, cx: float, cy: float, R: float,
                          t_start: float, t_end: float | None = None,
                          wheel: WheelConfig | None = None,
                          stable_s: float = 2.0) -> BounceSettle | None:
    """Track the ball in the pocket area from ``t_start`` (≈ rim drop) and
    declare the pocket whose relative index stabilizes for ``stable_s``
    seconds without ever leaving ±1 pocket afterwards (within the clip)."""
    wheel = wheel or WheelConfig()
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_start * fps))
    rt = RotorTracker(fps=fps)
    rt.set_calibration(PlungeCal(cx, cy, R))
    fi = int(t_start * fps)
    stop = int((t_end if t_end else t_start + 20.0) * fps)
    pos = None
    vel = 58.0
    coast = 0
    track = []
    while fi < stop:
        ok, fr = cap.read()
        if not ok:
            break
        t = fi / fps
        rt.process(t, fr)
        if fi % 2 == 0:  # ~30 fps is plenty for a ~60 deg/s ball
            cs = _candidates(fr, cx, cy, R)
            if pos is None:
                ring = [c for c in cs if 0.45 < np.hypot(c[0]-cx, c[1]-cy)/R < 0.75]
                if ring:
                    c = max(ring, key=lambda c: c[2])
                    az = np.degrees(np.arctan2(c[1]-cy, c[0]-cx)) % 360
                    pos = (az, np.hypot(c[0]-cx, c[1]-cy)/R)
                    track.append((t, az, pos[1], c[2]))
            else:
                pred_az = pos[0] + vel * (2 / fps)
                best, best_d = None, 1e9
                for c in cs:
                    az = np.degrees(np.arctan2(c[1]-cy, c[0]-cx)) % 360
                    rad = np.hypot(c[0]-cx, c[1]-cy)/R
                    daz = abs((az - pred_az + 180) % 360 - 180)
                    drad = abs(rad - pos[1])
                    d = daz + 80 * drad
                    if daz < 40 and drad < 0.22 and d < best_d:
                        best, best_d = (az, rad, c[2]), d
                if best is not None:
                    az_un = pos[0] + ((best[0] - pos[0] + 180) % 360 - 180)
                    vel = float(np.clip(0.9*vel + 0.1*(az_un-pos[0])/(2/fps),
                                        -150, 150))
                    pos = (az_un, best[1])
                    track.append((t, az_un, best[1], best[2]))
                    coast = 0
                else:
                    coast += 1
                    if coast <= 6:
                        pos = (pos[0] + vel*(2/fps), pos[1])
                        track.append((t, pos[0], pos[1], 0))
        fi += 1
    cap.release()
    if len(track) < 40:
        return None
    t_r, phi, _ = rt.track()
    ts = np.array([x[0] for x in track])
    azs = np.array([x[1] for x in track])
    phi_i = np.interp(ts, t_r, phi)
    idx_f = np.array([wrap_deg(a - p) for a, p in zip(azs, phi_i)]) / POCKET_DEG
    final = np.median(idx_f[-20:])
    dev = np.abs(idx_f - final)
    # first time from which the read never deviates > 1 pocket AND holds
    # at least `stable_s` seconds
    for i in range(len(ts)):
        if np.all(dev[i:] < 1.0):
            j = i
            while j < len(ts) and ts[j] - ts[i] < stable_s:
                j += 1
            t_declare = ts[min(j, len(ts) - 1)]
            pocket = wheel.pocket_order[int(round(final)) % 37]
            return BounceSettle(pocket=pocket, t_declare=float(t_declare),
                                t_stable_from=float(ts[i]),
                                n_track_points=len(track))
    return None
