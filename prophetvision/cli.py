"""Command-line interface.

  prophetvision demo [--save spin.avi]        synthetic end-to-end demo
  prophetvision analyze VIDEO [options]       predict from a real video
  prophetvision calibrate VIDEO               show detected wheel geometry
  prophetvision live VIDEO [options]          real-time session (SPEC F-H)
  prophetvision history VIDEO                 dump the OCR history banner
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _wheel(args):
    from .config import WheelConfig
    return WheelConfig(variant=args.wheel)


def cmd_demo(args) -> int:
    from .tracking import demo
    demo(save_avi=args.save, out_scatter=args.out_scatter)
    return 0


def cmd_analyze(args) -> int:
    import numpy as np

    from .tracking import TrackResult, analyze_video
    from .predict import LandingZonePredictor

    wheel = _wheel(args)
    center = tuple(float(v) for v in args.center.split(",")) if args.center else None
    result: TrackResult = analyze_video(
        args.video,
        max_seconds=args.max_seconds,
        center=center,
        radius=args.radius,
    )
    predictor = LandingZonePredictor(wheel=wheel)
    pred = predictor.predict(result)
    zone, mass = pred.best_zone(args.zone_width)
    print(json.dumps({
        "t_drop": round(pred.t_drop, 3),
        "t_impact": round(pred.t_impact, 3),
        "top_pocket": pred.top_pocket,
        "zone_width": args.zone_width,
        "zone": zone,
        "zone_probability": round(mass, 4),
        "impact_pocket_index": pred.impact_pocket_index,
        "n_ball_samples": len(result.theta_ball),
    }, indent=2))
    if args.out:
        np.savez_compressed(
            args.out,
            probabilities=pred.probabilities,
            pocket_order=np.array(wheel.pocket_order),
            t=result.t,
            theta_ball=result.theta_ball,
        )
        print(f"saved -> {args.out}", file=sys.stderr)
    return 0


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
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run the synthetic end-to-end demo")
    d.add_argument("--save", metavar="AVI", help="also save the synthetic video")
    d.add_argument("--out-scatter", metavar="JSON", help="scatter output path")
    d.set_defaults(fn=cmd_demo)

    a = sub.add_parser("analyze", help="predict the landing zone from a video")
    a.add_argument("video")
    a.add_argument("--wheel", choices=["european", "american"], default="european")
    a.add_argument("--max-seconds", type=float, default=None)
    a.add_argument("--center", help="manual cx,cy (pixels)")
    a.add_argument("--radius", type=float, help="manual rim radius (pixels)")
    a.add_argument("--zone-width", type=int, default=9)
    a.add_argument("--out", help="save probabilities NPZ")
    a.set_defaults(fn=cmd_analyze)

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