"""Polar-median ball tracker with Kalman fusion. SPEC.md section C.

Pipeline (per frame, streaming — never stores full frames):
  1. polar unwrap of the rim annulus [r_in, r_out] x R into
     (n_rad x n_bins) via a precomputed cv2.remap map,
  2. sliding temporal median background (~45 frames) subtraction — static
     reflections are absorbed into the background and never detected,
  3. detection = compact bright blob: local maxima of the radial-max band
     above max(absolute ~25/255, relative MAD) with parabolic sub-bin
     interpolation (< 0.5 deg at 720 bins),
  4. Kalman filter on state (theta_unwrapped, omega, alpha) with white-jerk
     process noise; prediction every frame, nearest-peak association with
     3-sigma innovation gating plus a hard 45 deg/frame jump rejection;
     dropouts <= 0.5 s are bridged by pure prediction, longer gaps cut the
     segment,
  5. static-blob rejection: a tentative track whose last ACCEPTED
     detections travel < static_travel_deg is killed within a few frames
     and its azimuth is blacklisted for re-init (a ball in the rim annulus
     ALWAYS moves fast; below ~rotor speed it falls). Established tracks
     that stall the same way are cut (ball dropped or static capture).

Track loss / drop event: a segment ends when the ball leaves the annulus
(it falls) or is lost for > 0.5 s.  ``drop_time`` exposes the last real
detection time of the most recent validated segment — this is the 'drop'
event consumed by live.py.

``track()`` returns (t, theta_unwrapped_deg) with segments concatenated and
separated by a NaN row; ``segments`` returns the list of per-segment
(t, theta) arrays.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import ObliqueCal, PlungeCal

__all__ = ["BallSample", "PolarMedianTracker", "BallTracker"]


@dataclass
class BallSample:
    t: float
    azimuth: float      # unwrapped degrees
    strength: float


class PolarMedianTracker:
    """See module docstring (SPEC.md section C)."""

    def __init__(self, cal: PlungeCal | ObliqueCal, fps: float,
                 r_in: float = 0.86, r_out: float = 1.00, n_bins: int = 720):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.cal = cal
        self.fps = float(fps)
        self.r_in = float(r_in)
        self.r_out = float(r_out)
        self.n_bins = int(n_bins)
        self.n_rad = 12
        self.deg_per_bin = 360.0 / self.n_bins

        # --- tunables -------------------------------------------------
        self.bg_window = 45               # sliding median background frames
        self.bg_warmup = 8                # min frames before detecting
        self.abs_thresh = 25.0            # absolute peak floor (per 255)
        self.rel_thresh = 6.0             # x robust (MAD) noise of the band
        self.max_peaks = 8                # candidate blobs per frame
        self.max_jump_deg = 45.0          # hard per-frame jump rejection
        self.max_gap_s = 0.5              # dropout bridging budget
        self.weak_thresh = 12.0           # track-guided dim-ball recovery
        self.weak_meas_sigma_deg = 1.2    # down-weight weak accepts
        self.max_speed_deg = 2000.0       # sanity clamp on omega
        self.min_travel_deg = 15.0        # segment validation
        self.min_segment_s = 0.2
        self.min_segment_n = 5
        self.static_confirm_n = 6         # frames to convict a static blob
        self.static_travel_deg = 3.5      # < this over confirm_n => static
        self.blacklist_s = 3.0            # re-init ban around a static blob
        self.blacklist_radius_deg = 10.0
        self.ghost_min_span_s = 0.6       # ghost conviction: min track span
        self.ghost_max_accept_ratio = 0.3  # < this accepted fraction => ghost
        self.meas_sigma_deg = 0.3         # sub-bin measurement noise
        self.jerk_var = 1.5e5             # (deg/s^3)^2 white-jerk intensity
        self.alpha_max = 150.0            # deg/s^2 physical clamp (ball ~15)

        # --- polar unwrap maps ----------------------------------------
        self._map_x, self._map_y = self._build_maps()

        # --- streaming state ------------------------------------------
        self._win: deque[np.ndarray] = deque(maxlen=self.bg_window)
        self._x: np.ndarray | None = None     # [theta, omega, alpha]
        self._P: np.ndarray | None = None
        self._last_t: float | None = None
        self._coast = 0.0
        self._n_accepted = 0
        self._n_processed = 0  # frames processed since track init
        self._acc_az: deque[tuple[float, float]] = deque(maxlen=64)
        self._cur: list[BallSample] = []
        self._segments: list[list[BallSample]] = []
        self._blacklist: list[tuple[float, float]] = []  # (az_deg, expiry_t)
        self._drop_time: float | None = None

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def _build_maps(self) -> tuple[np.ndarray, np.ndarray]:
        ang = (np.arange(self.n_bins) + 0.5) * 2.0 * np.pi / self.n_bins
        rad = np.linspace(self.r_in, self.r_out, self.n_rad)
        AA, RR = np.meshgrid(ang, rad)          # (n_rad, n_bins)
        cal = self.cal
        if isinstance(cal, ObliqueCal):
            # ellipse rim: canonical circle azimuth = parameter theta
            x0 = cal.a * RR * np.cos(AA)
            y0 = cal.b * RR * np.sin(AA)
            ca, sa = np.cos(np.radians(cal.angle)), np.sin(np.radians(cal.angle))
            xs = cal.cx + ca * x0 - sa * y0
            ys = cal.cy + sa * x0 + ca * y0
        else:
            # PlungeCal (circle)
            xs = cal.cx + cal.R * RR * np.cos(AA)
            ys = cal.cy + cal.R * RR * np.sin(AA)
        return xs.astype(np.float32), ys.astype(np.float32)

    def _unwrap(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return cv2.remap(gray, self._map_x, self._map_y,
                         cv2.INTER_LINEAR).astype(np.float32)

    # ------------------------------------------------------------------
    # detection
    # ------------------------------------------------------------------
    def _detect(self, u: np.ndarray
                ) -> tuple[list[tuple[float, float]],
                           list[tuple[float, float]], np.ndarray | None]:
        """Return (strong candidates, weak candidates, background).

        Candidates are (azimuth_deg_wrapped, strength) local maxima of the
        background-subtracted radial-max band with parabolic sub-bin
        interpolation.  Strong = above max(abs_thresh, relative MAD);
        weak = above weak_thresh only (used for track-guided recovery of a
        dim ball while coasting, never for track init).
        """
        self._win.append(u)
        if len(self._win) < self.bg_warmup:
            return [], [], None
        bg = np.median(np.stack(self._win), axis=0)
        band = (u - bg).max(axis=0)             # max over radius (blur streak)
        band = band - np.median(band)
        # 3-bin circular smoothing: the fast ball is a motion-blur streak
        # spread over several angular bins; smoothing recovers its peak
        # while narrow static spots gain less.
        band = (np.roll(band, 1) + band + np.roll(band, -1)) / 3.0
        pos = np.clip(band, 0.0, None)
        mad = 1.4826 * float(np.median(np.abs(band)))
        thr = max(self.abs_thresh, self.rel_thresh * mad)
        lo = min(thr, self.weak_thresh)
        if float(pos.max()) < lo:
            return [], [], bg
        local = (pos >= lo) & (pos > np.roll(pos, 1)) & \
            (pos >= np.roll(pos, -1))
        idx = np.flatnonzero(local)
        if idx.size == 0:                       # flat plateau: take argmax
            idx = np.array([int(np.argmax(pos))])
        idx = idx[np.argsort(pos[idx])[::-1]][: self.max_peaks]
        strong, weak = [], []
        for k in idx:
            y0, y1, y2 = float(pos[(k - 1) % self.n_bins]), float(pos[k]), \
                float(pos[(k + 1) % self.n_bins])
            den = y0 - 2.0 * y1 + y2
            delta = 0.5 * (y0 - y2) / den if den > 1e-6 else 0.0
            delta = float(np.clip(delta, -1.0, 1.0))
            az = ((k + delta + 0.5) * self.deg_per_bin) % 360.0
            (strong if y1 >= thr else weak).append((az, y1))
        return strong, weak, bg

    # ------------------------------------------------------------------
    # Kalman (theta_unwrapped, omega, alpha), white-jerk model
    # ------------------------------------------------------------------
    def _predict(self, dt: float) -> None:
        F = np.array([[1.0, dt, 0.5 * dt * dt],
                      [0.0, 1.0, dt],
                      [0.0, 0.0, 1.0]])
        q = self.jerk_var
        Q = q * np.array([[dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
                          [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
                          [dt**3 / 6.0, dt**2 / 2.0, dt]])
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q
        self._clamp_state()

    def _clamp_state(self) -> None:
        self._x[1] = float(np.clip(self._x[1],
                                   -self.max_speed_deg, self.max_speed_deg))
        self._x[2] = float(np.clip(self._x[2], -self.alpha_max, self.alpha_max))

    def _associate(self, cands: list[tuple[float, float]],
                   gate_sigma: float = 3.0,
                   meas_sigma: float | None = None
                   ) -> tuple[float, float] | None:
        """Nearest candidate inside the gate (gate_sigma x sigma, hard
        45 deg)."""
        best = None
        best_dz = None
        R = (meas_sigma if meas_sigma is not None
             else self.meas_sigma_deg)**2
        S = self._P[0, 0] + R
        gate = min(gate_sigma * float(np.sqrt(S)), self.max_jump_deg)
        for az, strength in cands:
            dz = (az - self._x[0] + 180.0) % 360.0 - 180.0
            if abs(dz) <= gate and (best_dz is None or abs(dz) < abs(best_dz)):
                best, best_dz = (az, strength), dz
        if best is None:
            return None
        K = self._P[:, 0] / S
        self._x = self._x + K * best_dz
        self._P = self._P - np.outer(K, self._P[0, :])
        self._clamp_state()
        return best

    # ------------------------------------------------------------------
    # segment bookkeeping
    # ------------------------------------------------------------------
    def _is_blacklisted(self, az: float, t: float) -> bool:
        self._blacklist = [(a, e) for a, e in self._blacklist if e > t]
        for a, _e in self._blacklist:
            if abs((az - a + 180.0) % 360.0 - 180.0) < self.blacklist_radius_deg:
                return True
        return False

    def _close_segment(self, reason: str) -> None:
        seg, self._cur = self._cur, []
        self._x = None
        self._P = None
        self._last_t = None
        self._coast = 0.0
        self._n_accepted = 0
        self._n_processed = 0
        self._acc_az.clear()
        if len(seg) >= self.min_segment_n:
            dur = seg[-1].t - seg[0].t
            travel = abs(seg[-1].azimuth - seg[0].azimuth)
            n_acc = sum(1 for s in seg if s.strength > 0.0)
            # require real detection support, not just coasted predictions
            if dur >= self.min_segment_s and travel >= self.min_travel_deg \
                    and n_acc >= max(self.static_confirm_n, len(seg) // 4):
                self._segments.append(seg)
                # ball still moving fast when the track ended => it dropped;
                # the drop instant is the last real detection of the segment
                if reason in ("lost", "slowed"):
                    for s in reversed(seg):
                        if s.strength > 0.0:
                            self._drop_time = s.t
                            break

    def _kill_static(self, t: float, blacklist: bool = True) -> None:
        """Current track is a static blob (or a ghost): drop it, optionally
        blacklist its azimuth."""
        if blacklist and self._cur:
            az = self._cur[-1].azimuth % 360.0
            self._blacklist.append((az, t + self.blacklist_s))
        self._cur = []
        self._x = None
        self._P = None
        self._last_t = None
        self._coast = 0.0
        self._n_accepted = 0
        self._acc_az.clear()
        self._n_processed = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def process(self, t: float, frame: np.ndarray) -> None:
        t = float(t)
        u = self._unwrap(frame)
        strong, weak, _bg = self._detect(u)

        if self._x is None:
            # no active track: init on strongest non-blacklisted candidate
            for az, strength in strong:
                if not self._is_blacklisted(az, t):
                    self._x = np.array([az, 0.0, 0.0])
                    self._P = np.diag([2.0**2, 200.0**2, 500.0**2])
                    self._last_t = t
                    self._n_accepted = 1
                    self._n_processed = 1
                    self._coast = 0.0
                    self._acc_az.clear()
                    self._acc_az.append((t, az))
                    self._cur = [BallSample(t, az, strength)]
                    break
            return

        dt = t - self._last_t if self._last_t is not None else 1.0 / self.fps
        if dt <= 0.0:
            dt = 1.0 / self.fps
        self._predict(dt)
        self._last_t = t

        hit = self._associate(strong) if strong else None
        if hit is None and weak and self._coast > 0.0 and \
                self._n_accepted >= self.static_confirm_n:
            # established track coasting: track-guided recovery of a dim
            # ball (motion blur) — weak peaks are gated tighter (2 sigma)
            # and down-weighted (they are much noisier than strong peaks)
            hit = self._associate(weak, gate_sigma=2.0,
                                  meas_sigma=self.weak_meas_sigma_deg)
        if hit is not None:
            az, strength = hit
            self._coast = 0.0
            self._n_accepted += 1
            self._acc_az.append((t, float(self._x[0])))
            self._cur.append(BallSample(t, float(self._x[0]), strength))
            # travel-based static conviction over the last ACCEPTED
            # detections spanning >= 3 frame intervals (duplicate frames
            # must not fake a stall): a rim ball NEVER crawls — kills
            # tentative locks on static reflections and cuts established
            # tracks that stall (ball dropped / static capture).
            if len(self._acc_az) >= self.static_confirm_n:
                t0, az0 = self._acc_az[-self.static_confirm_n]
                if (t - t0) >= 3.0 / self.fps and \
                        abs(self._acc_az[-1][1] - az0) < self.static_travel_deg:
                    if self._n_accepted <= self.static_confirm_n:
                        self._kill_static(t)
                    else:
                        self._close_segment("slowed")
        else:
            self._coast += dt
            if self._coast > self.max_gap_s:
                self._close_segment("lost")
            else:
                # bridge the dropout with the pure prediction
                self._cur.append(BallSample(t, float(self._x[0]), 0.0))

        # Ghost conviction: a track sustained mostly on coasted predictions
        # (accepted-detection ratio far below any real ball coverage, which
        # is >= 90 %) is a phantom lock — e.g. a compression-artifact blob
        # right after a camera cut.  Kill it (no azimuth blacklist: the
        # ghost's azimuth is meaningless and must not ban the real ball).
        if self._x is not None:
            self._n_processed += 1
            if (len(self._cur) >= 2
                    and self._cur[-1].t - self._cur[0].t
                    >= self.ghost_min_span_s
                    and self._n_processed >= 10
                    and self._n_accepted / self._n_processed
                    < self.ghost_max_accept_ratio):
                self._kill_static(t, blacklist=False)

    def _finalize(self) -> None:
        if self._x is not None or self._cur:
            self._close_segment("lost")

    @property
    def segments(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Validated segments as (t, theta_unwrapped_deg) arrays."""
        self._finalize()
        out = []
        for seg in self._segments:
            out.append((np.array([s.t for s in seg]),
                        np.array([s.azimuth for s in seg])))
        return out

    @property
    def drop_time(self) -> float | None:
        """t at which the ball left the rim annulus (end of last valid
        segment), or None if no validated segment has ended."""
        return self._drop_time

    def track(self) -> tuple[np.ndarray, np.ndarray]:
        """(t, theta_unwrapped_deg): validated segments concatenated with a
        NaN separator row between consecutive segments."""
        ts, ths = [], []
        for i, (st, sth) in enumerate(self.segments):
            if i:
                ts.append(np.array([np.nan]))
                ths.append(np.array([np.nan]))
            ts.append(st)
            ths.append(sth)
        if not ts:
            return np.empty(0), np.empty(0)
        return np.concatenate(ts), np.concatenate(ths)


class BallTracker:
    """Multi-shot facade (SPEC.md section C).

    ``set_shot`` must be called on each shot entry with its calibration
    (recalibration at every shot change). 'plunge' and 'oblique' shots are
    tracked with PolarMedianTracker (ellipse mapping for ObliqueCal, best
    effort); any other shot pauses tracking.  ``track()`` concatenates the
    segments of every shot tracker (NaN-separated).  ``drop_time`` exposes
    the most recent ball-drop time across trackers (live.py 'drop' event).
    """

    def __init__(self, fps: float):
        self.fps = float(fps)
        self._trackers: list[PolarMedianTracker] = []
        self._cur: PolarMedianTracker | None = None
        self._shot: str | None = None

    def set_shot(self, shot: str, cal=None) -> None:
        if shot == self._shot and cal is None:
            return
        self._shot = shot
        if cal is None or shot not in ("plunge", "oblique"):
            self._cur = None
            return
        if self._cur is not None:
            self._cur._finalize()
        tr = PolarMedianTracker(cal, self.fps)
        self._trackers.append(tr)
        self._cur = tr

    def process(self, t: float, frame: np.ndarray) -> None:
        if self._cur is not None:
            self._cur.process(t, frame)

    @property
    def segments(self) -> list[tuple[np.ndarray, np.ndarray]]:
        segs: list[tuple[np.ndarray, np.ndarray]] = []
        for tr in self._trackers:
            tr._finalize()
            segs.extend(tr.segments)
        return segs

    @property
    def current_segment(self) -> tuple[np.ndarray, np.ndarray,
                                       np.ndarray] | None:
        """In-progress (not yet validated/closed) segment of the active
        tracker as (t, theta_unwrapped_deg, strength) arrays — strength 0
        marks coasted Kalman predictions (no real detection) — or None when
        no track is active.  Read-only view for live prediction (SPEC F)."""
        tr = self._cur
        if tr is None or not tr._cur:
            return None
        return (np.array([s.t for s in tr._cur], dtype=float),
                np.array([s.azimuth for s in tr._cur], dtype=float),
                np.array([s.strength for s in tr._cur], dtype=float))

    @property
    def drop_time(self) -> float | None:
        drops = [tr.drop_time for tr in self._trackers
                 if tr.drop_time is not None]
        return max(drops) if drops else None

    def track(self) -> tuple[np.ndarray, np.ndarray]:
        ts, ths = [], []
        for st, sth in self.segments:
            ts.append(st)
            ths.append(sth)
            ts.append(np.array([np.nan]))
            ths.append(np.array([np.nan]))
        if not ts:
            return np.empty(0), np.empty(0)
        return np.concatenate(ts[:-1]), np.concatenate(ths[:-1])
