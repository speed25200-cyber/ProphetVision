"""Command-line interface.

  prophetvision demo [--save spin.avi]        synthetic end-to-end demo
  prophetvision analyze VIDEO [options]       predict from a real spin video
  prophetvision audit VIDEO                   feasibility verdict for a stream
  prophetvision calibrate VIDEO               show detected wheel geometry
  prophetvision live VIDEO [options]          real-time session (SPEC F-H)
  prophetvision history VIDEO                 dump the OCR history banner
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import WheelConfig, AMERICAN_ORDER
from .pipeline import analyze_frames, read_video
from .predict import LandingZonePredictor
from .scatter import ScatterModel
from .synthetic import SyntheticSpin, write_video
from .tracking import Calibration


def _wheel(args) -> WheelConfig:
    w = WheelConfig()
    if getattr(args, "wheel", "european") == "american":
        w.pocket_order = list(AMERICAN_ORDER)
    return w


def _load_predictor(args) -> LandingZonePredictor:
    p = LandingZonePredictor(wheel=_wheel(args))
    path = getattr(args, "scatter_file", None)
    if path and os.path.exists(path):
        with open(path) as f:
            p.scatter = ScatterModel.from_json(f.read())
    return p


def cmd_demo(args) -> int:
    predictor = None
    # Optional warm-up: observe some spins first so the scatter model is
    # learned rather than uniform, as it would be in real use.
    for k in range(args.train):
        s = SyntheticSpin(seed=10_000 + args.seed * 100 + k)
        fr, _, tru = s.simulate()
        p, _, predictor = analyze_frames(
            fr, s.fps, predictor=predictor,
            cutoff_seconds=tru.t_drop - args.lead)
        predictor.observe_outcome(tru.final_pocket)
    spin = SyntheticSpin(seed=args.seed)
    frames, times, truth = spin.simulate()
    if args.save:
        write_video(args.save, frames, spin.fps)
        print(f"synthetic spin written to {args.save}")
    cutoff = truth.t_drop - args.lead
    pred, track, _ = analyze_frames(frames, spin.fps, predictor=predictor,
                                    cutoff_seconds=cutoff)
    s = pred.summary(zone_width=args.zone_width)
    n = spin.wheel.n_pockets
    err = min((pred.impact_pocket_index - truth.impact_index) % n,
              (truth.impact_index - pred.impact_pocket_index) % n)
    print(json.dumps({
        "prediction_made_at_s": round(cutoff, 2),
        "ball_dropped_at_s": round(truth.t_drop, 2),
        "true_final_pocket": truth.final_pocket,
        "impact_pocket_error_pockets": err,
        "zone_hit": truth.final_pocket in s["zone"],
        **s,
    }, indent=2))
    return 0


def cmd_analyze(args) -> int:
    frames, fps = read_video(args.video, max_seconds=args.max_seconds)
    predictor = _load_predictor(args)
    cal = None
    if args.center and args.radius:
        cx, cy = (float(v) for v in args.center.split(","))
        cal = Calibration(cx=cx, cy=cy, radius=float(args.radius))
    pred, track, predictor = analyze_frames(
        frames, fps, predictor=predictor, cutoff_seconds=args.cutoff,
        calibration=cal)
    print(json.dumps(pred.summary(zone_width=args.zone_width), indent=2))
    if args.outcome is not None:
        predictor.observe_outcome(args.outcome)
        if args.scatter_file:
            with open(args.scatter_file, "w") as f:
                f.write(predictor.scatter.to_json())
            print(f"scatter model updated -> {args.scatter_file} "
                  f"({predictor.scatter.n_observations():.0f} observations)")
    return 0


def cmd_audit(args) -> int:
    from .feasibility import audit
    cal = None
    if args.center and args.radius:
        cx, cy = (float(v) for v in args.center.split(","))
        cal = Calibration(cx=cx, cy=cy, radius=float(args.radius))
    rep = audit(args.video, wheel=_wheel(args), start_s=args.start,
                end_s=args.end, calibration=cal)
    print(json.dumps(rep.summary(), indent=2))
    return 0 if rep.predictable else 1


def cmd_calibrate(args) -> int:
    frames, fps = read_video(args.video, max_seconds=2.0)
    cal = Calibration.detect(frames)
    print(json.dumps({"cx": cal.cx, "cy": cal.cy, "radius": cal.radius,
                      "fps": fps, "frames_read": len(frames)}, indent=2))
    return 0


def cmd_live(args) -> int:
    """Full real-time session (SPEC.md sections F-H).  Streams the video
    (never buffers it) and prints the session summary as JSON."""
    from .live import LiveEngine
    engine = LiveEngine(wheel=_wheel(args), realtime=args.realtime,
                        zone_width=args.zone_width,
                        max_seconds=args.max_seconds)
    if args.verbose:
        def on_event(ev):
            print("event " + json.dumps(ev, default=str), file=sys.stderr)
    else:
        on_event = None
    report = engine.run(args.video, on_event=on_event)
    if args.json:
        report.save(args.json)
        print(f"session JSON -> {args.json}", file=sys.stderr)
    if args.report:
        from .dashboard import render_spin_report
        frames_dir = os.path.splitext(args.report)[0] + "_frames"
        render_spin_report(report, args.video, args.report, frames_dir)
        print(f"rapport HTML -> {args.report} (frames: {frames_dir})",
              file=sys.stderr)
    print(json.dumps(report.summary(), indent=2, ensure_ascii=False))
    return 0


def cmd_history(args) -> int:
    """Dump the OCR history banner over time (validation tool, SPEC H).
    Streams; prints one JSON line per banner change."""
    from .chronology import Chronology
    from .streaming import FrameSource, dedup_times
    chrono = Chronology(banner_period=args.period)
    src = FrameSource(args.video)
    for t, frame in dedup_times(src):
        if args.max_seconds is not None and t > args.max_seconds:
            break
        for ev in chrono.process(t, frame):
            if ev.kind == "banner":
                print(json.dumps({"t": round(t, 2),
                                  "numbers": ev.detail["numbers"]}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="prophetvision", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="synthetic end-to-end demo")
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--train", type=int, default=0,
                   help="number of warm-up spins to learn the scatter model")
    d.add_argument("--lead", type=float, default=2.0,
                   help="seconds before ball-drop at which to predict")
    d.add_argument("--zone-width", type=int, default=9)
    d.add_argument("--save", help="write the synthetic spin video here")
    d.set_defaults(fn=cmd_demo)

    a = sub.add_parser("analyze", help="predict from a real spin video")
    a.add_argument("video")
    a.add_argument("--wheel", choices=["european", "american"],
                   default="european")
    a.add_argument("--cutoff", type=float, default=None,
                   help="only use the first N seconds (live-prediction mode)")
    a.add_argument("--max-seconds", type=float, default=None)
    a.add_argument("--zone-width", type=int, default=9)
    a.add_argument("--center", help="wheel center override 'cx,cy' (pixels)")
    a.add_argument("--radius", help="wheel radius override (pixels)")
    a.add_argument("--scatter-file", help="JSON file persisting the learned "
                   "scatter model between spins")
    a.add_argument("--outcome", type=int, default=None,
                   help="true final pocket, to update the scatter model")
    a.set_defaults(fn=cmd_analyze)

    au = sub.add_parser("audit", help="can this video support prediction at all?")
    au.add_argument("video")
    au.add_argument("--start", type=float, default=0.0)
    au.add_argument("--end", type=float, default=None)
    au.add_argument("--wheel", choices=["european", "american"],
                    default="european")
    au.add_argument("--center", help="wheel center override 'cx,cy' (pixels)")
    au.add_argument("--radius", help="wheel radius override (pixels)")
    au.set_defaults(fn=cmd_audit)

    c = sub.add_parser("calibrate", help="detect wheel center/radius")
    c.add_argument("video")
    c.set_defaults(fn=cmd_calibrate)

    li = sub.add_parser("live", help="real-time session: tracking, "
                        "predictions, chronology, HTML/JSON report")
    li.add_argument("video")
    li.add_argument("--wheel", choices=["european", "american"],
                    default="european")
    li.add_argument("--report", help="write the HTML spin report here")
    li.add_argument("--json", help="write the full session JSON here")
    li.add_argument("--zone-width", type=int, default=9)
    li.add_argument("--realtime", action="store_true",
                    help="do not consume the stream faster than real time")
    li.add_argument("--max-seconds", type=float, default=None)
    li.add_argument("--verbose", action="store_true",
                    help="print events to stderr as they happen")
    li.set_defaults(fn=cmd_live)

    hi = sub.add_parser("history", help="dump the OCR history banner")
    hi.add_argument("video")
    hi.add_argument("--period", type=float, default=0.5,
                    help="seconds between banner reads")
    hi.add_argument("--max-seconds", type=float, default=None)
    hi.set_defaults(fn=cmd_history)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
