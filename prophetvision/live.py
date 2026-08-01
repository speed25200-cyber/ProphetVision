"""Real-time session engine. SPEC.md section F (+ NOTES v2.1).

Pipeline per session (all streaming, one frame at a time):

    FrameSource -> dedup_times -> ShotSegmenter
        - every deduplicated frame feeds Chronology (countdown, bets_close,
          result, banner OCR),
        - on 'plunge' shots only: PlungeCal.detect (first 5 plunge frames,
          default (610, 419, 345) on failure) -> BallTracker + RotorTracker,
        - as soon as the in-progress ball segment spans >= 1.2 s of arc and
          >= 30 samples: LOCAL LINEAR drop estimate (NOTES v2.1: on this
          regime the c0+c2*omega^2 closed form extrapolates absurd drop
          times; omega ~ 110->55 deg/s is quasi-linear) -> scatter-corrected
          pocket distribution -> 'prediction' event, refreshed ~2x/s,
        - on ball drop: 'drop' event + online learning of omega_drop
          (measured ~50-60 deg/s here, NEVER the v1 default 640),
        - on OCR 'result': observe_outcome() (the scatter absorbs the raw
          impact offset, calibrated or not — logged per spin for honesty).

Honesty (SPEC tests note): the ball is launched AFTER bets close on this
stream; the report measures and displays launch - bets_close and the real
prediction lead time rather than claiming in-window betting feasibility.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from .balltrack import BallTracker
from .calibration import PlungeCal
from .chronology import Chronology
from .config import WheelConfig
from .geometry import wrap_deg
from .physics import BallDecayModel, RotorModel
from .predict import LandingZonePredictor, Prediction
from .rotortrack import RotorTracker
from .streaming import (FrameSource, ShotSegmenter, _frame_diff,
                        _small_gray)

__all__ = ["LiveEngine", "SessionReport", "linear_kinematics"]

#: Measured plunge calibration fallback (SPEC, NOTES v2.1).
DEFAULT_PLUNGE_CAL = (610.0, 419.0, 345.0)


def adaptive_zone(probs: np.ndarray, wheel: WheelConfig,
                  target: float = 0.67, max_width: int = 25
                  ) -> dict:
    """Smallest contiguous arc of pockets reaching ``target`` probability
    mass (the scatter on this wheel is REAL and WIDE — a fixed 9-pocket
    zone is often too optimistic; honesty = size the zone from the learned
    distribution)."""
    n = wheel.n_pockets
    p = np.asarray(probs, dtype=float)
    best = None
    for w in range(1, min(max_width, n) + 1):
        best_start, best_mass = 0, -1.0
        for s in range(n):
            mass = float(sum(p[(s + j) % n] for j in range(w)))
            if mass > best_mass:
                best_start, best_mass = s, mass
        if best is None or best_mass > best[2]:
            best = (best_start, w, best_mass)
        if best_mass >= target:
            best = (best_start, w, best_mass)
            break
    start, w, mass = best
    return {
        "width": w,
        "zone": [int(wheel.pocket_order[(start + j) % n]) for j in range(w)],
        "zone_p": round(float(mass), 4),
    }


# ----------------------------------------------------------------------
# Local linear drop estimate (NOTES v2.1 finding #1)
# ----------------------------------------------------------------------
def linear_kinematics(t: np.ndarray, theta: np.ndarray,
                      window_s: float = 1.5,
                      strength: np.ndarray | None = None) -> dict | None:
    """Fit local constant-deceleration kinematics on the LAST ``window_s``
    seconds of a ball segment.

    ``strength`` (optional) marks real detections (> 0) vs coasted Kalman
    predictions (== 0); the fit uses real samples only.  Returns
    dict(t_now, theta_now, omega, alpha, rms) with signed omega (deg/s) and
    alpha (deg/s^2, roughly -sign(omega)*|decel|), or None when the window
    has too little support.  On the spin tail (omega ~ 110->55 deg/s over
    ~4 s) the motion is quasi-linear, so this local fit is robust — unlike
    the global c0+c2*omega^2 closed form, which extrapolates absurd drop
    times on this regime.
    """
    t = np.asarray(t, dtype=float)
    th = np.asarray(theta, dtype=float)
    if len(t) < 6:
        return None
    t_now = float(t[-1])
    m = t >= t_now - window_s
    if strength is not None:
        m = m & (np.asarray(strength, dtype=float) > 0.0)
    if m.sum() < 8:
        return None
    tm = t[m] - t_now
    c, res, *_ = np.linalg.lstsq(
        np.column_stack([tm * tm, tm, np.ones_like(tm)]), th[m], rcond=None)
    pred = c[0] * tm * tm + c[1] * tm + c[2]
    rms = float(np.sqrt(np.mean((pred - th[m]) ** 2)))
    return {
        "t_now": t_now,
        "theta_now": float(c[2]),  # smoothed endpoint
        "omega": float(c[1]),
        "alpha": float(2.0 * c[0]),
        "rms": rms,
        "t_last_real": (float(t[m][-1]) if strength is not None else t_now),
    }


@dataclass
class SessionReport:
    """Result of a LiveEngine session (SPEC.md section F)."""

    video: str = ""
    spins: list = field(default_factory=list)
    n_frames: int = 0
    wall_s: float = 0.0
    omega_drop_final: float | None = None
    #: Learned scatter snapshot (counts include the unit uniform prior).
    scatter: dict | None = None

    @property
    def pipeline_fps(self) -> float:
        return self.n_frames / self.wall_s if self.wall_s > 0 else 0.0

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        preds = [p for s in self.spins for p in s.get("predictions", [])]
        leads = [p["t_drop_est"] - p["t_emit"] for p in preds]
        first_leads = [s["predictions"][0]["t_drop_est"]
                       - s["predictions"][0]["t_emit"]
                       for s in self.spins if s.get("predictions")]
        spin_leads = [s["lead_s"] for s in self.spins
                      if s.get("lead_s") is not None]
        hits = [s for s in self.spins if s.get("hit")]
        hits_ad = [s for s in self.spins if s.get("hit_adaptive")]
        hits_raw = [s for s in self.spins if s.get("hit_raw")]
        results = [s["result"] for s in self.spins
                   if s.get("result") is not None]
        ad_widths = [s["predictions"][-1]["zone_adaptive"]["width"]
                     for s in self.spins if s.get("predictions")]
        gaps = [s["feasibility"]["launch_minus_bets_close_s"]
                for s in self.spins
                if s.get("feasibility", {}).get("launch_minus_bets_close_s")
                is not None]
        feasible = bool(gaps) and all(g <= 0 for g in gaps)
        verdict = (
            "paris ouverts au lancement : prediction jouable sur ce flux"
            if feasible else
            "bille lancee APRES la fermeture des paris "
            f"(gap moyen {np.mean(gaps):.1f} s) : prediction non jouable en "
            "mise sur ce flux — valeur demontree par la precision mesuree "
            "et l'avance prediction->chute"
            if gaps else "chronologie paris/lancement incomplete")
        return {
            "video": self.video,
            "n_spins": len(self.spins),
            "spins_with_prediction": sum(1 for s in self.spins
                                         if s.get("predictions")),
            "n_predictions": len(preds),
            "results": results,
            "hits": len(hits),
            "hit_rate": (len(hits) / len(results)) if results else None,
            "hits_adaptive_zone": len(hits_ad),
            "hits_raw_zone": len(hits_raw),
            "mean_adaptive_zone_width": (float(np.mean(ad_widths))
                                         if ad_widths else None),
            "mean_lead_s": (float(np.mean(spin_leads))
                            if spin_leads else None),
            "mean_first_prediction_lead_s": (float(np.mean(first_leads))
                                             if first_leads else None),
            "max_prediction_lead_s": (float(np.max(leads))
                                      if leads else None),
            "mean_zone_p": (float(np.mean(
                [s["predictions"][-1]["zone_p"] for s in self.spins
                 if s.get("predictions")])) if preds else None),
            "omega_drop_final": self.omega_drop_final,
            "n_frames": self.n_frames,
            "wall_s": round(self.wall_s, 2),
            "pipeline_fps": round(self.pipeline_fps, 1),
            "feasibility": {
                "launch_minus_bets_close_s": gaps,
                "betting_feasible": feasible,
                "verdict": verdict,
            },
        }

    # ------------------------------------------------------------------
    def to_json(self) -> str:
        def default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)
        return json.dumps({
            "video": self.video,
            "n_frames": self.n_frames,
            "wall_s": round(self.wall_s, 3),
            "omega_drop_final": self.omega_drop_final,
            "scatter": self.scatter,
            "summary": self.summary(),
            "spins": self.spins,
        }, indent=2, default=default)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())


# ----------------------------------------------------------------------
class LiveEngine:
    """Full real-time pipeline (SPEC.md section F).  See module docstring."""

    #: Minimum in-progress segment arc (s) before the first prediction.
    min_arc_s = 1.2
    #: Minimum in-progress segment sample count before predicting.
    min_samples = 30
    #: Minimum wall-of-video seconds between two predictions (~2 Hz).
    refresh_s = 0.5
    #: Do not emit a prediction whose estimated drop is closer than this
    #: (a "prediction" 0.2 s before the fall is neither useful nor honest).
    min_lead_s = 0.5
    #: Window (s, from the segment end) of the local linear kinematics fit.
    fit_window_s = 1.5
    #: Bounds for the online-learned omega_drop (deg/s).
    omega_drop_bounds = (30.0, 120.0)
    #: Settle delay after a plunge cut before the trackers are started:
    #: the first ~0.2-0.3 s of a cut carry compression/fade artifacts that
    #: spawn phantom blobs (measured on hq_060_100).
    plunge_settle_s = 0.15

    def __init__(self, wheel: WheelConfig | None = None,
                 realtime: bool = False, zone_width: int = 9,
                 max_seconds: float | None = None):
        self.wheel = wheel or WheelConfig()
        self.realtime = bool(realtime)
        self.zone_width = int(zone_width)
        self.max_seconds = max_seconds
        self.predictor = LandingZonePredictor(wheel=self.wheel)
        # NOTES v2.1 priors: ball and rotor are quasi-synchronous on the
        # spin tail, so the ball drops at ~rotor speed and falls almost
        # straight down.  omega_drop is refined online at every drop.
        self.predictor.omega_drop = 55.0
        self.predictor.fall_time = 0.2
        self.predictor.fall_travel_deg = 20.0

    # ------------------------------------------------------------------
    @staticmethod
    def _tune_for_resolution(ball: BallTracker, cal: PlungeCal) -> None:
        """Scale the tracker detection floors to the wheel size (measured:
        the ball blob peak drops with resolution; reference R=345 px at
        1206x910 with abs_thresh=25 — at R=273 / 954x720 the usable floor
        is ~9-16).  The relative MAD term of the detector is unchanged."""
        tr = ball._cur
        if tr is None:
            return
        s = float(np.clip(cal.R / 345.0, 0.5, 1.2))
        tr.abs_thresh = max(9.0, 25.0 * s * s)
        tr.weak_thresh = max(5.0, 12.0 * s * s)

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------
    def _predict(self, t_seg: np.ndarray, th_seg: np.ndarray,
                 strength_seg: np.ndarray, rotor: RotorTracker,
                 t_plunge_start: float) -> dict | None:
        """Linear drop estimate + scatter-corrected pocket distribution.

        Returns a JSON-friendly prediction dict (and registers the
        underlying Prediction for observe_outcome), or None.
        """
        kin = linear_kinematics(t_seg, th_seg, window_s=self.fit_window_s,
                                strength=strength_seg)
        if kin is None:
            return None
        # Recency: no real detection for > 0.2 s => the track is coasting
        # (ball lost / dropped); never predict from a phantom tail.
        if kin["t_now"] - kin["t_last_real"] > 0.2:
            return None
        omega, alpha = kin["omega"], kin["alpha"]
        if abs(omega) < 5.0:
            return None
        sign = 1.0 if omega >= 0 else -1.0
        decel = -sign * alpha
        # Degenerate local fit (flat / accelerating): the drop extrapolation
        # is unreliable (measured decel on this wheel is ~10-40 deg/s^2).
        if decel < 5.0:
            return None
        dt = max(abs(omega) - self.predictor.omega_drop, 0.0) / decel
        t_now = kin["t_now"]
        t_drop = t_now + dt
        if not np.isfinite(t_drop) or dt > 12.0:
            return None
        theta_drop = (kin["theta_now"] + omega * dt + 0.5 * alpha * dt * dt)
        t_impact = t_drop + self.predictor.fall_time
        theta_impact = theta_drop + sign * self.predictor.fall_travel_deg

        # --- rotor phase at impact
        rt, rphi, is_abs = rotor.track()
        m = rt >= t_plunge_start - 0.1
        if m.sum() < 5:
            m = np.ones(len(rt), dtype=bool)
        if m.sum() < 3:
            return None
        rotor_model = RotorModel.fit(rt[m], rphi[m])
        if is_abs:
            phi_zero = rotor_model.phi_at(t_impact)
        else:
            phi_zero = (self.predictor.rotor_zero_azimuth
                        + (rotor_model.phi_at(t_impact)
                           - rotor_model.phi_at(rt[m][0])))
        rel = wrap_deg(theta_impact - phi_zero)
        n = self.wheel.n_pockets
        impact_idx = int(round(rel / (360.0 / n))) % n

        # --- scatter-corrected distribution (absorbs the raw offset once
        # calibrated; offsets are in the ball's direction of travel)
        offsets = self.predictor.scatter.distribution()
        probs = np.zeros(n)
        for off in range(n):
            probs[(impact_idx + int(sign) * off) % n] += offsets[off]
        probs /= probs.sum()

        # A BallDecayModel surrogate matching the local linear kinematics
        # (c2 -> 0 limit of the closed form), so observe_outcome() can read
        # the travel direction and the dashboard can overlay the fit.
        ball_model = BallDecayModel(
            c0=decel, c2=1e-9, t_ref=t_now, omega_ref=abs(omega),
            theta_ref=kin["theta_now"], direction=sign)
        pred = Prediction(
            probabilities=probs, wheel=self.wheel,
            impact_pocket_index=impact_idx,
            t_drop=t_drop, t_impact=t_impact,
            ball_model=ball_model, rotor_model=rotor_model)
        self.predictor._last_prediction = pred
        zone, mass = pred.best_zone(self.zone_width)
        half = self.zone_width // 2
        raw_zone = [int(self.wheel.pocket_order[(impact_idx + j) % n])
                    for j in range(-half, half + 1)]
        return {
            "t_emit": round(t_now, 3),
            "t_drop_est": round(t_drop, 3),
            "zone": [int(z) for z in zone],
            "zone_p": round(float(mass), 4),
            "zone_adaptive": adaptive_zone(probs, self.wheel),
            "raw_zone": raw_zone,
            "top_pocket": int(pred.top_pocket),
            "impact_pocket_index": int(impact_idx),
            "arc_used_s": round(float(t_seg[-1] - t_seg[0]), 3),
            "omega_now": round(abs(omega), 2),
            "theta_impact_deg": round(float(wrap_deg(theta_impact)), 2),
            "probs": [round(float(p), 5) for p in probs],
        }

    # ------------------------------------------------------------------
    # spin bookkeeping
    # ------------------------------------------------------------------
    def _new_spin(self, t: float, pending_events: list) -> dict:
        return {
            "index": 0,  # fixed on append
            "t_plunge_start": round(float(t), 3),
            "t_plunge_end": None,
            "calibration": None,
            "events": list(pending_events),
            "predictions": [],
            "drop": None,
            "result": None,
            "hit": None,
            "lead_s": None,
            "track_stats": None,
            "feasibility": {"launch_minus_bets_close_s": None},
            "series": None,
        }

    @staticmethod
    def _segment_at(ball: BallTracker, t_drop: float):
        """Validated (t, theta) segment ending at t_drop, or None."""
        for st, sth in ball.segments:
            if len(st) >= 6 and abs(st[-1] - t_drop) < 1.0:
                return st, sth
        return None

    def _omega_at_segment_end(self, ball: BallTracker,
                              t_drop: float) -> float | None:
        """|omega| (deg/s) smoothed over the last ~1 s of the segment that
        ends at t_drop — the online omega_drop measurement (NOTES v2.1)."""
        seg = self._segment_at(ball, t_drop)
        if seg is None:
            return None
        st, sth = seg
        # Fit on samples BEFORE t_drop: the segment tail past t_drop is
        # coasted Kalman prediction (its extrapolated speed is fictitious).
        m = (st <= t_drop + 0.02) & (st >= t_drop - 1.0)
        if m.sum() < 4:
            m = st <= t_drop + 0.02
        if m.sum() < 3:
            return None
        c = np.polyfit(st[m] - t_drop, sth[m], 2)
        return float(abs(c[1]))

    def _finalize_spin(self, spin: dict, ball: BallTracker,
                       rotor: RotorTracker, t_end: float,
                       n_plunge_frames: int) -> None:
        spin["t_plunge_end"] = round(float(t_end), 3)
        t0, t1 = spin["t_plunge_start"], t_end
        # --- track stats + series over the spin window
        segs = [(st, sth) for st, sth in ball.segments
                if len(st) and st[-1] >= t0 - 0.5 and st[0] <= t1 + 0.5]
        arc = float(sum(abs(sth[-1] - sth[0]) for st, sth in segs))
        v0 = None
        n_samp = int(sum(len(st) for st, _ in segs))
        if segs and len(segs[0][0]) >= 6:
            st, sth = segs[0]
            m = st <= st[0] + 1.0
            if m.sum() >= 4:
                c = np.polyfit(st[m] - st[0], sth[m], 2)
                v0 = float(abs(c[1]))
        span = (max((st[-1] for st, _ in segs), default=t0)
                - min((st[0] for st, _ in segs), default=t0))
        # coverage = sampled fraction of the ball-segment time span (the
        # ball window is only ~4 s inside an ~18 s plunge shot)
        eff_fps = n_plunge_frames / max(t1 - t0 - self.plunge_settle_s, 1e-9)
        coverage = (n_samp / max(span * eff_fps, 1.0)) if span > 0 else None
        spin["track_stats"] = {
            "arc_deg": round(arc, 1),
            "v0": (round(v0, 1) if v0 is not None else None),
            "coverage": (round(min(coverage, 1.0), 3)
                         if coverage is not None else None),
            "n_samples": n_samp,
        }
        # --- downsampled series for the dashboard curves/annotations
        bt, bth, bom = [], [], []
        for st, sth in segs:
            if len(st) >= 6:
                om = np.gradient(sth, st)
                k = max(3, min(9, len(st) // 4 * 2 + 1))
                ker = np.ones(k) / k
                om = np.convolve(om, ker, mode="same")
            else:
                om = np.full(len(st), np.nan)
            bt.extend(st.tolist())
            bth.extend(sth.tolist())
            bom.extend([float(v) for v in om])
        rt, rphi, _abs = rotor.track()
        m = (rt >= t0 - 0.5) & (rt <= t1 + 0.5)
        rt, rphi = rt[m], rphi[m]
        rom = np.gradient(rphi, rt) if len(rt) >= 2 else np.full(len(rt), np.nan)

        def ds(arr, cap=240):
            arr = np.asarray(arr, dtype=float)
            if len(arr) <= cap:
                return [round(float(v), 3) for v in arr]
            idx = np.linspace(0, len(arr) - 1, cap).astype(int)
            return [round(float(v), 3) for v in arr[idx]]

        spin["series"] = {
            "t_ball": ds(bt), "theta_ball": ds(bth), "omega_ball": ds(bom),
            "t_rotor": ds(rt), "phi_rotor": ds(rphi), "omega_rotor": ds(rom),
        }
        # --- lead of the LAST prediction (avance reelle annoncee)
        if spin["predictions"]:
            last = spin["predictions"][-1]
            spin["lead_s"] = round(last["t_drop_est"] - last["t_emit"], 3)
        # --- real lead of every prediction vs the (final) measured drop
        if spin["drop"] is not None:
            for p in spin["predictions"]:
                p["lead_actual_s"] = round(spin["drop"]["t"] - p["t_emit"], 3)
        # --- feasibility: launch (= plunge start) vs bets close
        bc = [e for e in spin["events"] if e["kind"] == "bets_close"]
        if bc:
            spin["feasibility"]["launch_minus_bets_close_s"] = round(
                t0 - min(e["t"] for e in bc), 3)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self, video_path: str, on_event=None) -> SessionReport:
        src = FrameSource(video_path)
        # banner_period: the history strip is secondary (the result marker
        # is the primary truth); 1.5 s checks keep per-frame cost low.
        chrono = Chronology(banner_period=1.5)
        # vote_every=5: cuts still re-classify every frame during the
        # hysteresis window; steady-state Hough votes are throttled (perf).
        segmenter = ShotSegmenter(vote_every=5)
        ball = BallTracker(src.fps)
        rotor = RotorTracker(src.fps, wheel=self.wheel)

        report = SessionReport(video=video_path)
        spins: list[dict] = report.spins
        pending_events: list[dict] = []   # pre-spin chronology events

        def emit(kind: str, t: float, **kw):
            if on_event is not None:
                on_event({"kind": kind, "t": round(float(t), 3), **kw})

        spin: dict | None = None
        shot_prev: str | None = None
        cal: PlungeCal | None = None
        cal_tries = 0
        n_plunge_frames = 0
        last_pred_t = -1e9
        drop_done_t: float | None = None
        tracking_started = False

        wall0 = time.perf_counter()
        vid_t0 = None
        n_frames = 0
        # Inlined dedup_times loop (same semantics, SPEC section A) so the
        # 300x226 grayscale is shared with the ShotSegmenter instead of
        # being computed twice per frame (~3 ms/frame saved).
        prev_small = None
        for _idx, t, frame in src:
            small = _small_gray(frame)
            if prev_small is not None and \
                    _frame_diff(small, prev_small) < 0.08:
                continue  # duplicated container frame: drop it
            prev_small = small
            t = float(t)
            if vid_t0 is None:
                vid_t0 = t
            if self.max_seconds is not None and t - vid_t0 > self.max_seconds:
                break
            if self.realtime:
                lag = (t - vid_t0) - (time.perf_counter() - wall0)
                if lag > 0:
                    time.sleep(lag)
            n_frames += 1

            # --- chronology on EVERY deduplicated frame
            for ev in chrono.process(t, frame):
                e = {"kind": ev.kind, "t": round(float(ev.t), 3),
                     "detail": dict(ev.detail)}
                target = spin
                if ev.kind == "result":
                    # attach to the active spin, else the most recent spin
                    # still waiting for its result
                    if target is None:
                        for s in reversed(spins):
                            if s["result"] is None:
                                target = s
                                break
                if target is not None:
                    target["events"].append(e)
                else:
                    pending_events.append(e)
                emit(ev.kind, ev.t, **ev.detail)
                if ev.kind == "result" and target is not None:
                    number = int(ev.detail["number"])
                    target["result"] = number
                    self.predictor.observe_outcome(number)
                    preds = target["predictions"]
                    if preds:
                        last = preds[-1]
                        target["hit"] = bool(number in last["zone"])
                        target["hit_adaptive"] = bool(
                            number in last["zone_adaptive"]["zone"])
                        target["hit_raw"] = bool(number in last["raw_zone"])
                        target["impact_pocket_index"] = \
                            last["impact_pocket_index"]
                        idx_res = self.wheel.index_of(number) \
                            if number in self.wheel.pocket_order else None
                        target["offset_raw"] = (
                            (idx_res - last["impact_pocket_index"])
                            % self.wheel.n_pockets
                            if idx_res is not None else None)

            # --- shot segmentation
            shot = segmenter.process(frame, small=small)

            if shot == "plunge" and shot_prev != "plunge":
                # spin starts: attach pending pre-spin events
                spin = self._new_spin(t, pending_events)
                spin["index"] = len(spins)
                spin["events"].append({"kind": "launch", "t": round(t, 3),
                                       "detail": {"shot": "plunge"}})
                spins.append(spin)
                pending_events = []
                cal, cal_tries = None, 0
                n_plunge_frames = 0
                last_pred_t = -1e9
                tracking_started = False
                emit("launch", t)

            if shot == "plunge" and spin is not None:
                if cal is None:
                    cal = PlungeCal.detect(frame)
                    cal_tries += 1
                    if cal is None and cal_tries >= 5:
                        cal = PlungeCal(*DEFAULT_PLUNGE_CAL)
                    if cal is not None:
                        spin["calibration"] = {
                            "cx": round(cal.cx, 1), "cy": round(cal.cy, 1),
                            "R": round(cal.R, 1),
                            "detected": cal_tries < 5
                            or cal.R != DEFAULT_PLUNGE_CAL[2]}
                settled = (t - spin["t_plunge_start"] >= self.plunge_settle_s)
                if cal is not None and settled:
                    if not tracking_started:
                        ball.set_shot("plunge", cal)
                        self._tune_for_resolution(ball, cal)
                        rotor.set_calibration(cal)
                        tracking_started = True
                    ball.process(t, frame)
                    rotor.process(t, frame)
                    n_plunge_frames += 1

                    # --- prediction state machine
                    seg = ball.current_segment
                    if (seg is not None
                            and t - last_pred_t >= self.refresh_s):
                        st, sth, sst = seg
                        if (len(st) >= self.min_samples
                                and st[-1] - st[0] >= self.min_arc_s):
                            pred = self._predict(st, sth, sst, rotor,
                                                 spin["t_plunge_start"])
                            if (pred is not None
                                    and pred["t_drop_est"] - t
                                    >= self.min_lead_s):
                                spin["predictions"].append(pred)
                                last_pred_t = t
                                emit("prediction", t, **pred)

                    # --- drop event + omega_drop online learning
                    dt_ = ball.drop_time
                    if dt_ is not None and dt_ != drop_done_t:
                        drop_done_t = dt_
                        seg = self._segment_at(ball, dt_)
                        arc = (float(abs(seg[1][-1] - seg[1][0]))
                               if seg is not None else 0.0)
                        om_end = self._omega_at_segment_end(ball, dt_)
                        cand = {"t": round(float(dt_), 3),
                                "omega_end": (round(om_end, 2)
                                              if om_end else None),
                                "arc_deg": round(arc, 1)}
                        # The true drop is the end of the MAIN (longest-arc)
                        # ball segment; short phantom segments ending early
                        # must not replace it nor poison omega_drop.
                        cur = spin["drop"]
                        if cur is None or arc > cur.get("arc_deg", 0.0):
                            spin["drop"] = cand
                            # Learn only from a real ball track (>= 100 deg
                            # arc) with a PHYSICALLY plausible end speed:
                            # the ball falls near rotor speed (~50-65 deg/s
                            # here); noisy end-fits (short/coarsed tracks)
                            # must not poison omega_drop.
                            if (om_end is not None and arc >= 100.0
                                    and 40.0 <= om_end <= 90.0):
                                lo, hi = self.omega_drop_bounds
                                self.predictor.omega_drop = float(
                                    np.clip(om_end, lo, hi))
                        emit("drop", dt_, omega_end=om_end)

            elif shot != "plunge" and shot_prev == "plunge" \
                    and spin is not None:
                ball.set_shot(shot)  # pauses tracking (oblique abandoned)
                self._finalize_spin(spin, ball, rotor, t, n_plunge_frames)
                spin = None

            shot_prev = shot

        if spin is not None:
            ball.set_shot("other")
            self._finalize_spin(spin, ball, rotor,
                                spin["events"][-1]["t"]
                                if spin["events"] else 0.0,
                                n_plunge_frames)
        report.n_frames = n_frames
        report.wall_s = time.perf_counter() - wall0
        report.omega_drop_final = round(self.predictor.omega_drop, 2)
        report.scatter = {
            # counts include the unit uniform prior of the ScatterModel
            "n_observations": round(
                self.predictor.scatter.n_observations() - 1.0, 3),
            "sigma_pockets": self.predictor.scatter.sigma,
            "counts": [round(float(c), 4)
                       for c in self.predictor.scatter.counts],
            "distribution": [round(float(p), 5) for p in
                             self.predictor.scatter.distribution()],
        }
        return report
