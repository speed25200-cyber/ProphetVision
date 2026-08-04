"""Robust measurement of the impact index — where the ball leaves the rim.

This quantity is the hinge of the whole accuracy chain, and measuring it
naively is unreliable: two careful implementations disagreed by 14 and 16
pockets on two of five spins (while agreeing to 0.54 on a third). What was
being reported as "bounce dispersion" was largely this measurement noise. The
three failure modes, all observed, are handled here.

**Camera cuts inside the window.** A cut put a different framing — wider,
wheel off-centre, multiplier bubbles — in the first half of one spin's
"top-down" window. The circle calibration does not fit it and the rotor tracker
locked onto a static overlay, reading 0 deg/s. :func:`find_cuts` locates cuts by
frame-to-frame difference so the measurement can start after the last one.

**Fixed artefacts masquerading as the ball.** A reflection sitting still in the
image trivially satisfies "relative index decreasing", because the rotor moves
underneath it. Radius and confidence cuts cannot separate the two; the *lab
frame* can. Undo the rotor rotation and a fixed artefact has zero speed, while
the ball is still running several pockets per second and — this is the decisive
part — running *against* the rotor. :func:`measure_impact` therefore converts
each candidate arc's rate back to the lab frame and demands both a direction
opposite the rotor and a minimum speed. Note that a ratio test on the relative
rate would not do: near contact the ball's relative rate is only about 1.6x the
rotor's, so any threshold loose enough to keep the ball also keeps artefacts.

**The track outliving the impact.** The clearest signal in the data is what
happens at the end: the ball holds a steady lab-frame speed all the way round
the rim (-92 deg/s, spin 1) and then loses it in about 0.15 s while its radius
keeps falling — it has struck a deflector and is dropping. The detector happily
follows the falling, nearly stationary blob for another third of a second, and
in that third of a second the rotor turns almost four pockets. Reading the
impact off the last sample of the track therefore biases it by that much. The
impact is the last sample that is still *running*, which is what
:func:`measure_impact` returns.

**Choosing between several plausible arcs.** Taking the latest one that passes
the motion tests is not enough: on spin A a nine-sample fragment at r=0.78,
half a second after the ball had already landed, beat the eighty-seven-sample
arc that was the ball. The arc that matters is the one that *ends by leaving
the rim* — the ball spirals down, so its track finishes low, whereas a track
still at r=1.0 is the ball mid-flight and a short fragment is noise. Requiring
the descent and then preferring the longest arc picks the right one on every
spin here.

**Several objects at once.** The detector returns five to ten boxes per frame,
so "the detections between t and t+dt" is not a trajectory — on one spin the
ball and a fixed highlight at azimuth 224 are both present, frame for frame,
for a full second. Splitting a *cloud* into time-contiguous runs silently
interleaves them. :func:`link_tracks` therefore assigns detections to tracks by
predicted azimuth first, and only then is each track measured on its own.
Bridging a gap of a second would let a track re-acquire on the wrong object, so
tracks are cut at gaps and the samples and span of the one used are reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

POCKET_DEG = 360.0 / 37.0


def find_cuts(path: str, t0: float, t1: float, time_offset: float = 0.0,
              threshold: float = 8.0, size=(300, 226)) -> list[float]:
    """Times of hard camera cuts in [t0, t1] (clip time), offset applied."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
    stop = int(t1 * fps)
    prev = None
    cuts = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if idx >= stop:
                break
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                               size).astype(np.float32)
            if prev is not None and float(np.abs(small - prev).mean()) > threshold:
                cuts.append(time_offset + idx / fps)
            prev = small
    finally:
        cap.release()
    return cuts


@dataclass
class ImpactResult:
    """Where and when the ball left the rim, with the evidence for it."""

    index: float                 # relative index (rotor frame) at rim exit
    t: float
    azimuth: float               # lab-frame azimuth of the exit: which deflector
    confidence: float
    radius: float
    arc_samples: int             # length of the running part of the track
    arc_span_s: float
    rel_rate_pockets_s: float    # index rate in the rotor frame
    lab_rate_pockets_s: float    # the same arc seen from the ground
    rotor_pockets_s: float
    n_runs: int                  # tracks that passed the tests
    min_lab_rate: float = 3.0
    exit_lag_s: float = 0.0      # track time discarded after the ball stopped
    descent: float = 0.0         # radius lost over the whole track

    @property
    def trustworthy(self) -> bool:
        """A measurement worth feeding into the bounce statistics.

        Two independent things have to hold: the arc must *move* like a ball
        (lab-frame speed, against the rotor) and it must *descend* like one.
        The descent is the sharper of the two on real footage — the ball's
        terminal arc loses about a fifth of the bowl radius while every fixed
        highlight measured here moved by less than 0.02. It is also what lets a
        short arc count: on spin B the camera cut to the top-down view only
        0.37 s before the ball left the rim, so the arc is a quarter of a
        second long, and a length threshold would have thrown away a
        measurement that is in fact unambiguous.
        """
        return (self.arc_samples >= 12 and self.descent >= 0.05
                and abs(self.lab_rate_pockets_s) >= self.min_lab_rate
                and self.lab_rate_pockets_s * self.rotor_pockets_s < 0.0
                and self.confidence >= 0.50)


def _unwrap_index(idx: np.ndarray) -> np.ndarray:
    """Unwrap a pocket-index series through the 0/37 seam."""
    return np.degrees(np.unwrap(np.radians(idx * POCKET_DEG))) / POCKET_DEG


def _local_rate(t: np.ndarray, y: np.ndarray, half_window: float = 0.15,
                min_pts: int = 5) -> np.ndarray:
    """Slope of ``y`` against ``t`` in a window around each sample.

    A two-point difference is far too noisy on this detector (a degree of box
    jitter at 50 Hz is 50 deg/s of apparent speed), and a single fit over the
    whole track would smear the very transition being looked for.
    """
    n = len(t)
    v = np.zeros(n)
    for i in range(n):
        lo = int(np.searchsorted(t, t[i] - half_window))
        hi = int(np.searchsorted(t, t[i] + half_window, side="right"))
        if hi - lo < min_pts:
            hi = min(n, max(hi, lo + min_pts))
            lo = max(0, hi - min_pts)
        if hi - lo < 2:
            continue
        v[i] = float(np.polyfit(t[lo:hi] - t[i], y[lo:hi], 1)[0])
    return v


def link_tracks(rows, max_gap_s: float = 0.12, tol_deg: float = 14.0,
                dr_tol: float = 0.07, min_samples: int = 6) -> list[np.ndarray]:
    """Group detections into tracks by continuity of azimuth.

    Rows are ``[t, az_deg, r_frac, conf, index]``. Each track is extended to the
    detection at the next instant closest to where its own motion says it should
    be, so a fast ball and a stationary highlight sitting on the same rim end up
    in different tracks instead of being averaged into one meaningless series.
    The velocity estimate is smoothed because the detector jitters by a degree
    or so frame to frame and an unsmoothed estimate lets a track wander off.
    """
    d = np.asarray(rows, dtype=float)
    if len(d) == 0:
        return []
    d = d[np.argsort(d[:, 0])]
    tracks: list[dict] = []
    for t in np.unique(d[:, 0]):
        here = d[d[:, 0] == t]
        taken = np.zeros(len(here), dtype=bool)
        for tr in tracks:
            dt = t - tr["t"]
            if dt <= 0 or dt > max_gap_s:
                continue
            pred = tr["az"] + tr["v"] * dt
            err = np.abs((here[:, 1] - pred + 180.0) % 360.0 - 180.0)
            ok = (~taken) & (err <= tol_deg + abs(tr["v"]) * dt * 0.25) \
                & (np.abs(here[:, 2] - tr["r"]) <= dr_tol)
            if not ok.any():
                continue
            j = int(np.where(ok, err, np.inf).argmin())
            taken[j] = True
            step = (here[j, 1] - tr["az"] + 180.0) % 360.0 - 180.0
            v = step / dt
            tr["v"] = v if tr["n"] < 3 else 0.6 * tr["v"] + 0.4 * v
            tr["t"], tr["az"], tr["r"] = t, here[j, 1], here[j, 2]
            tr["n"] += 1
            tr["rows"].append(here[j])
        for j in np.where(~taken)[0]:
            tracks.append({"t": t, "az": here[j, 1], "r": here[j, 2],
                           "v": 0.0, "n": 1, "rows": [here[j]]})
    out = [np.array(tr["rows"]) for tr in tracks if tr["n"] >= min_samples]
    return sorted(out, key=lambda a: a[-1, 0])


def measure_impact(detections, rotor_rate_pockets_s: float,
                   r_rim: float = 0.78, max_gap_s: float = 0.12,
                   min_lab_rate: float = 3.0, min_samples: int = 6,
                   half_window: float = 0.15, r_exit: float = 0.88,
                   t_min: float | None = None) -> ImpactResult | None:
    """Impact index from detections already expressed in the rotor frame.

    ``detections`` is an (N, 5) array of ``[t, azimuth_deg, r_frac, confidence,
    relative_index]``, in any order. ``rotor_rate_pockets_s`` is the rotor's own
    signed rate in the lab frame, in pockets per second, in the same sense as
    the azimuth used to build ``relative_index``.

    Returns the *latest* contiguous rim arc whose lab-frame motion is opposite
    the rotor and fast enough to be a rolling ball, or None if none qualifies.
    """
    d = np.asarray(detections, dtype=float)
    if d.ndim != 2 or d.shape[1] < 5:
        raise ValueError("detections must be an (N, 5) array "
                         "[t, az, r, conf, index]")
    if t_min is not None:
        d = d[d[:, 0] >= t_min]
    if len(d) < min_samples:
        return None
    d = d[np.argsort(d[:, 0])]
    rim = d[d[:, 2] >= r_rim]
    if len(rim) < min_samples:
        return None
    runs = link_tracks(rim, max_gap_s=max_gap_s, min_samples=min_samples)

    best, n_ok = None, 0
    for track in runs:
        if len(track) < min_samples:
            continue
        if track[-1, 2] > r_exit:
            continue          # still high on the rim: this is not the end
        t = track[:, 0]
        lab_v = _local_rate(t, np.degrees(np.unwrap(np.radians(track[:, 1]))),
                            half_window) / POCKET_DEG
        # A fixed artefact has zero lab-frame speed; the ball still runs, and
        # runs against the rotor. Where that stops being true, so does the rim.
        running = ((lab_v * rotor_rate_pockets_s < 0.0)
                   & (np.abs(lab_v) >= min_lab_rate))
        if not running.any():
            continue
        end = int(np.where(running)[0][-1])
        run = track[:end + 1]
        if len(run) < min_samples:
            continue
        span = float(run[-1, 0] - run[0, 0])
        if span <= 0:
            continue
        idx = _unwrap_index(run[:, 4])
        rel_rate = float(np.polyfit(run[:, 0] - run[0, 0], idx, 1)[0])
        lab_rate = rel_rate + rotor_rate_pockets_s
        if lab_rate * rotor_rate_pockets_s >= 0.0 or abs(lab_rate) < min_lab_rate:
            continue
        n_ok += 1
        lag = float(track[-1, 0] - run[-1, 0])
        drop = float(track[0, 2] - track[-1, 2])
        key = (len(run), run[-1, 0])
        if best is None or key > (len(best[0]), best[0][-1, 0]):
            best = (run, rel_rate, lab_rate, span, lag, drop)
    if best is None:
        return None
    run, rel_rate, lab_rate, span, lag, drop = best
    last = run[-1]
    return ImpactResult(index=float(last[4]) % 37.0, t=float(last[0]),
                        azimuth=float(last[1]) % 360.0,
                        confidence=float(last[3]), radius=float(last[2]),
                        arc_samples=int(len(run)), arc_span_s=span,
                        rel_rate_pockets_s=rel_rate,
                        lab_rate_pockets_s=lab_rate,
                        rotor_pockets_s=float(rotor_rate_pockets_s),
                        n_runs=n_ok, min_lab_rate=min_lab_rate,
                        exit_lag_s=lag, descent=drop)


def speed_at_radius(detections, rotor_rate_pockets_s: float,
                    r_target: float = 0.95, r_rim: float = 0.78,
                    half_window: float = 0.20,
                    **kwargs) -> tuple[float, float, float] | None:
    """Ball speed, time and rotor-frame index where it crosses ``r_target``.

    This is the measurement that makes the drop time predictable. The ball
    spirals inward as it slows, and the radius it sits at is fixed by the
    balance of gravity against the bowl's slope — so the speed at a given
    radius is a property of the wheel, not of the spin. Measured on the
    reference footage it is 93.7 +- 1.1 deg/s at r = 0.95, a spread of 1.2%
    across four spins, which is far tighter than anything about the rim exit
    itself. Extrapolating the decay law down to *this* speed rather than to a
    guessed one is what brought the predicted crossing to within 0.4 s.

    Returns ``(speed_deg_s, t, index)``, or None if the tracked arc never
    reaches that radius (spin B: the camera cut to the top-down view when the
    ball was already at r = 0.89).
    """
    res = measure_impact(detections, rotor_rate_pockets_s, r_rim=r_rim,
                         half_window=half_window, **kwargs)
    if res is None:
        return None
    d = np.asarray(detections, dtype=float)
    rim = d[d[:, 2] >= r_rim]
    track = None
    for a in link_tracks(rim):
        if a[0, 0] <= res.t <= a[-1, 0]:
            track = a[a[:, 0] <= res.t]
    if track is None or len(track) < 3:
        return None
    r = track[:, 2]
    if not (r.min() <= r_target <= r.max()):
        return None
    v = _local_rate(track[:, 0],
                    np.degrees(np.unwrap(np.radians(track[:, 1]))),
                    half_window)
    k = int(np.argmin(np.abs(r - r_target)))
    return float(v[k]), float(track[k, 0]), float(track[k, 4]) % 37.0


def transfer_point(detections, rotor_rate_pockets_s, decay,
                   w_target: float = 93.6, r_rim: float = 0.78,
                   **kwargs) -> tuple[float, float] | None:
    """Time and rotor-frame index where the ball's speed crosses ``w_target``.

    This is the point the side-view prediction aims at, so it has to be
    measured well: near the transfer the deceleration is only ~20 deg/s^2, and
    the ball and rotor close on each other at ~18 pockets/s, so a few deg/s of
    slope error moves the crossing a quarter of a second and the index four
    pockets. Reading it off a local slope cost about that much. The terminal
    arc is dense and contiguous, so instead the decay law — one free parameter,
    the coefficients being properties of the wheel — is fitted to the whole of
    it and the crossing solved for in closed form.

    Returns ``(t, index)``, or None if the arc never brackets that speed.
    """
    from .earlyside import fit_speed

    res = measure_impact(detections, rotor_rate_pockets_s, r_rim=r_rim,
                         **kwargs)
    if res is None:
        return None
    d = np.asarray(detections, dtype=float)
    track = None
    for a in link_tracks(d[d[:, 2] >= r_rim]):
        if a[0, 0] <= res.t <= a[-1, 0]:
            track = a[a[:, 0] <= res.t]
    if track is None or len(track) < 6:
        return None
    t = track[:, 0]
    az = np.degrees(np.unwrap(np.radians(track[:, 1])))
    w0 = fit_speed(t, az, decay, omega_grid=np.arange(60.0, 220.0, 0.25))[1]
    t_x = float(t[0] + decay.time_to(w0, w_target))
    if not (t[0] - 0.60 <= t_x <= t[-1] + 0.30):
        return None
    idx = _unwrap_index(track[:, 4])
    slope = (idx[1] - idx[0]) / (t[1] - t[0])
    index = (float(np.interp(t_x, t, idx,
                             left=idx[0] + (t_x - t[0]) * slope)) % 37.0)
    return t_x, index


def bounce_spread(impacts, results, wheel=None) -> dict:
    """Anchor-free dispersion of the bounce, in pockets.

    ``impacts`` are measured impact indices and ``results`` the pocket numbers
    the spins actually paid. The unknown offset between the vision index and
    the wheel's numbering is a constant, so it shifts every ``v_i =
    index(result_i) - impact_i`` by the same amount and cancels out of their
    spread. That is what makes this measurable from a handful of spins without
    ever calibrating the offset.
    """
    from .config import WheelConfig

    wheel = wheel or WheelConfig()
    n = wheel.n_pockets
    v = np.array([wheel.index_of(int(p)) - float(i)
                  for i, p in zip(impacts, results)], dtype=float)
    ang = np.radians(v * (360.0 / n))
    c, s = np.cos(ang).mean(), np.sin(ang).mean()
    r = float(np.hypot(c, s))
    mean = float((np.degrees(np.arctan2(s, c)) / (360.0 / n)) % n)
    r_c = min(max(r, 1e-12), 1.0 - 1e-12)
    sigma = float(np.degrees(np.sqrt(-2.0 * np.log(r_c))) / (360.0 / n))
    # Rayleigh test: is the concentration more than chance for this many spins?
    m = len(v)
    z = m * r * r
    p = float(np.exp(-z) * (1.0 + (2.0 * z - z * z) / (4.0 * m)))
    return {"v": v, "mean_pockets": mean, "sigma_pockets": sigma,
            "resultant": r, "n": m, "rayleigh_p": min(max(p, 0.0), 1.0)}
