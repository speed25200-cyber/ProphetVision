"""HTML session/spin report. SPEC.md section G.

Self-contained dark report (inline CSS, inline SVG, JPEG frames saved in
``frames_dir`` and referenced relatively): per spin — wheel with per-pocket
probabilities, predicted zone arc, top pocket and real result markers,
omega(t) curves for ball and rotor with the drop point, chronology table,
honest feasibility banner (the ball is launched AFTER bets close on this
stream), hit/miss verdicts (raw 9-pocket zone, scatter-corrected zone,
adaptive-width zone) and annotated frames.  Session header aggregates
spins, hits, real prediction lead and the learned scatter distribution.
"""
from __future__ import annotations

import html
import math
import os

import cv2
import numpy as np

from .config import WheelConfig
from .streaming import FrameSource

# ---------------------------------------------------------------------------
# palette (elegant dark: near-black warm background, gold/amber accents,
# cream text — no flashy gradients)
# ---------------------------------------------------------------------------
_BG = "#12100e"
_CARD = "#1a1713"
_EDGE = "#2c2620"
_TEXT = "#e8dfc9"
_DIM = "#9a8f79"
_GOLD = "#c9a24a"
_AMBER = "#e0a82e"
_RED = "#c8543e"
_GREEN = "#7ba05b"
_TEAL = "#6f9a8d"

CSS = f"""
body {{ background:{_BG}; color:{_TEXT}; margin:0;
  font-family:'Segoe UI', 'Helvetica Neue', Arial, sans-serif; }}
.wrap {{ max-width:1060px; margin:0 auto; padding:28px 20px 60px; }}
h1 {{ font-size:26px; font-weight:600; color:{_GOLD}; margin:0 0 4px;
  letter-spacing:0.5px; }}
h2 {{ font-size:19px; color:{_GOLD}; border-bottom:1px solid {_EDGE};
  padding-bottom:6px; margin:34px 0 14px; }}
h3 {{ font-size:15px; color:{_TEXT}; margin:18px 0 8px; }}
.sub {{ color:{_DIM}; font-size:13px; margin-bottom:18px; }}
.cards {{ display:flex; flex-wrap:wrap; gap:10px; margin:14px 0; }}
.card {{ background:{_CARD}; border:1px solid {_EDGE}; border-radius:8px;
  padding:10px 16px; min-width:120px; }}
.card .v {{ font-size:22px; font-weight:600; color:{_AMBER}; }}
.card .k {{ font-size:11px; color:{_DIM}; text-transform:uppercase;
  letter-spacing:0.8px; }}
.panel {{ background:{_CARD}; border:1px solid {_EDGE}; border-radius:10px;
  padding:16px 18px; margin:12px 0; }}
table {{ border-collapse:collapse; font-size:13px; }}
td, th {{ padding:4px 14px 4px 0; color:{_TEXT}; vertical-align:top; }}
th {{ color:{_DIM}; font-weight:500; text-transform:uppercase;
  font-size:11px; letter-spacing:0.6px; }}
.banner {{ border-radius:8px; padding:12px 16px; margin:12px 0;
  font-size:14px; line-height:1.5; }}
.banner.warn {{ background:#2a1f14; border:1px solid #6b4d1f;
  color:{_AMBER}; }}
.banner.ok {{ background:#16211a; border:1px solid #33502e;
  color:{_GREEN}; }}
.hit {{ color:{_GREEN}; font-weight:600; }}
.miss {{ color:{_RED}; font-weight:600; }}
.dim {{ color:{_DIM}; }}
img.frame {{ border:1px solid {_EDGE}; border-radius:8px; margin:6px 8px 6px 0;
  max-width:480px; }}
.grid {{ display:flex; flex-wrap:wrap; gap:18px; }}
.legal {{ color:{_DIM}; font-size:11px; margin-top:40px;
  border-top:1px solid {_EDGE}; padding-top:12px; }}
"""


def _esc(s) -> str:
    return html.escape(str(s))


def _polar(cx, cy, r, ang_deg):
    a = math.radians(ang_deg)
    return cx + r * math.cos(a), cy + r * math.sin(a)


def _wedge(cx, cy, r_in, r_out, a0, a1):
    """SVG path of an annular sector (angles in degrees, 0 = +x, cw)."""
    large = 1 if (a1 - a0) % 360 > 180 else 0
    x0, y0 = _polar(cx, cy, r_out, a0)
    x1, y1 = _polar(cx, cy, r_out, a1)
    x2, y2 = _polar(cx, cy, r_in, a1)
    x3, y3 = _polar(cx, cy, r_in, a0)
    return (f"M{x0:.1f},{y0:.1f} A{r_out:.1f},{r_out:.1f} 0 {large} 1 "
            f"{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f} "
            f"A{r_in:.1f},{r_in:.1f} 0 {large} 0 {x3:.1f},{y3:.1f} Z")


def _prob_color(p, p_max):
    """Dark bronze -> amber scale by probability."""
    f = 0.0 if p_max <= 0 else min(1.0, p / p_max)
    f = f ** 0.6
    r = int(0x24 + (0xE0 - 0x24) * f)
    g = int(0x1F + (0xA8 - 0x1F) * f)
    b = int(0x18 + (0x2E - 0x18) * f)
    return f"#{r:02x}{g:02x}{b:02x}"


def wheel_svg(wheel: WheelConfig, probs, zone, zone_adaptive, top_pocket,
              result, size=340):
    """SVG of the wheel: pockets colored by probability, predicted zone
    arcs (9-pocket + adaptive), top pocket and real result marked."""
    n = wheel.n_pockets
    cx = cy = size / 2
    r_out = size * 0.46
    r_in = size * 0.30
    step = 360.0 / n
    p_max = max(probs) if probs else 0.0
    parts = [f'<svg width="{size}" height="{size}" viewBox="0 0 {size} '
             f'{size}" role="img">']
    for i, num in enumerate(wheel.pocket_order):
        a0 = -90 + i * step - step / 2
        a1 = a0 + step
        fill = _prob_color(probs[i] if probs else 0.0, p_max)
        parts.append(f'<path d="{_wedge(cx, cy, r_in, r_out, a0, a1 - 0.6)}" '
                     f'fill="{fill}" stroke="{_BG}" stroke-width="1"/>')
        tx, ty = _polar(cx, cy, (r_in + r_out) / 2, a0 + step / 2)
        parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="8.5" '
                     f'fill="{_BG}" text-anchor="middle" '
                     f'dominant-baseline="central" font-weight="700">'
                     f'{num}</text>')

    def zone_arc(numbers, r, color, width, dash=""):
        idx = [wheel.pocket_order.index(z) for z in numbers
               if z in wheel.pocket_order]
        if not idx:
            return
        # group consecutive indices (circular)
        idx.sort()
        groups = [[idx[0]]]
        for a, b in zip(idx, idx[1:]):
            (groups[-1].append(b) if b == a + 1 else groups.append([b]))
        if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == n - 1:
            groups[0] = groups.pop() + groups[0]
        for g in groups:
            a0 = -90 + g[0] * step - step / 2
            a1 = -90 + (g[-1] + 1) * step - step / 2
            x0, y0 = _polar(cx, cy, r, a0)
            x1, y1 = _polar(cx, cy, r, a1)
            large = 1 if (a1 - a0) > 180 else 0
            d = (f"M{x0:.1f},{y0:.1f} A{r:.1f},{r:.1f} 0 {large} 1 "
                 f"{x1:.1f},{y1:.1f}")
            parts.append(f'<path d="{d}" fill="none" stroke="{color}" '
                         f'stroke-width="{width}" stroke-linecap="round" '
                         f'{dash}/>')

    if zone_adaptive:
        zone_arc(zone_adaptive["zone"], r_out + 9, _TEAL, 5,
                 'stroke-dasharray="1 0" opacity="0.9"')
    if zone:
        zone_arc(zone, r_out + 16, _GOLD, 4)
    if top_pocket in wheel.pocket_order:
        i = wheel.pocket_order.index(top_pocket)
        tx, ty = _polar(cx, cy, r_in - 12, -90 + i * step + step / 2)
        parts.append(f'<path d="M{tx:.1f},{ty - 6:.1f} l5,9 l-10,0 Z" '
                     f'fill="{_AMBER}"/>')
    if result is not None and result in wheel.pocket_order:
        i = wheel.pocket_order.index(result)
        tx, ty = _polar(cx, cy, (r_in + r_out) / 2, -90 + i * step + step / 2)
        parts.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="7" fill="none" '
                     f'stroke="{_RED}" stroke-width="3"/>')
    parts.append("</svg>")
    return "".join(parts)


def omega_svg(spin, width=560, height=190):
    """SVG polyline chart of ball and rotor angular speed with drop."""
    ser = spin.get("series") or {}
    tb, ob = ser.get("t_ball") or [], ser.get("omega_ball") or []
    tr, orr = ser.get("t_rotor") or [], ser.get("omega_rotor") or []
    if not tb and not tr:
        return '<p class="dim">pas de piste</p>'
    ts = [x for x in list(tb) + list(tr) if x is not None]
    oms = [abs(x) for x in list(ob) + list(orr)
           if x is not None and math.isfinite(x)]
    if not ts or not oms:
        return '<p class="dim">pas de piste</p>'
    t0, t1 = min(ts), max(ts)
    om_max = max(oms) * 1.08
    pad_l, pad_b, pad_t = 44, 26, 10
    pw, ph = width - pad_l - 8, height - pad_b - pad_t

    def xy(t, om):
        x = pad_l + (t - t0) / max(t1 - t0, 1e-9) * pw
        y = pad_t + (1 - abs(om) / om_max) * ph
        return f"{x:.1f},{y:.1f}"

    parts = [f'<svg width="{width}" height="{height}" '
             f'style="background:{_BG};border:1px solid {_EDGE};'
             f'border-radius:8px">']
    for f in (0.25, 0.5, 0.75, 1.0):
        y = pad_t + (1 - f) * ph
        parts.append(f'<line x1="{pad_l}" y1="{y:.0f}" x2="{pad_l + pw:.0f}" '
                     f'y2="{y:.0f}" stroke="{_EDGE}" stroke-width="0.6"/>')
        parts.append(f'<text x="{pad_l - 5}" y="{y:.0f}" font-size="9" '
                     f'fill="{_DIM}" text-anchor="end" '
                     f'dominant-baseline="central">'
                     f'{om_max * f:.0f}</text>')
    for t in np.linspace(t0, t1, 6):
        x = pad_l + (t - t0) / max(t1 - t0, 1e-9) * pw
        parts.append(f'<text x="{x:.0f}" y="{height - 8}" font-size="9" '
                     f'fill="{_DIM}" text-anchor="middle">{t:.1f}s</text>')
    if tr and orr:
        pts = " ".join(xy(t, o) for t, o in zip(tr, orr)
                       if o is not None and math.isfinite(o))
        parts.append(f'<polyline points="{pts}" fill="none" '
                     f'stroke="{_TEAL}" stroke-width="1.6"/>')
    if tb and ob:
        pts = " ".join(xy(t, o) for t, o in zip(tb, ob)
                       if o is not None and math.isfinite(o))
        parts.append(f'<polyline points="{pts}" fill="none" '
                     f'stroke="{_AMBER}" stroke-width="1.8"/>')
    drop = spin.get("drop")
    if drop and drop.get("omega_end"):
        x, y = xy(drop["t"], drop["omega_end"]).split(",")
        parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{_RED}"/>')
        parts.append(f'<text x="{float(x) - 8:.1f}" y="{float(y) - 9:.1f}" '
                     f'font-size="10" fill="{_RED}" text-anchor="end">'
                     f'chute</text>')
    parts.append(f'<text x="{pad_l + 6}" y="16" font-size="10" '
                 f'fill="{_AMBER}">&#969; bille</text>')
    parts.append(f'<text x="{pad_l + 70}" y="16" font-size="10" '
                 f'fill="{_TEAL}">&#969; rotor</text>')
    parts.append("</svg>")
    return "".join(parts)


def scatter_svg(report, width=560, height=120):
    """Bar chart of the learned scatter distribution (offset pockets)."""
    sc = getattr(report, "scatter", None) or {}
    dist = sc.get("distribution")
    if not dist:
        return ""
    n = len(dist)
    k = np.arange(n)
    offs = np.minimum(k, n - k) * np.where(k <= n // 2, 1, -1)
    order = np.argsort(offs)
    pad = 24
    bw = (width - 2 * pad) / n
    parts = [f'<svg width="{width}" height="{height}" '
             f'style="background:{_BG};border:1px solid {_EDGE};'
             f'border-radius:8px">']
    for i in order:
        h = (height - 44) * dist[i] / max(dist)
        x = pad + offs[i] * bw + (width - 2 * pad) / 2 - bw / 2
        parts.append(f'<rect x="{x:.1f}" y="{height - 26 - h:.1f}" '
                     f'width="{bw * 0.8:.1f}" height="{h:.1f}" '
                     f'fill="{_GOLD}" opacity="0.85"/>')
        if offs[i] % 3 == 0:
            parts.append(f'<text x="{x + bw * 0.4:.1f}" y="{height - 12}" '
                         f'font-size="8" fill="{_DIM}" text-anchor="middle">'
                         f'{int(offs[i])}</text>')
    parts.append(f'<text x="{pad}" y="14" font-size="10" fill="{_DIM}">'
                 f'distribution de scatter apprise (offset poches, '
                 f'{sc.get("n_observations", 0):.0f} observations)</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# frame annotation
# ---------------------------------------------------------------------------
def _interp(xs, ys, x):
    if not xs:
        return None
    return float(np.interp(x, np.asarray(xs, dtype=float),
                           np.asarray(ys, dtype=float)))


def annotate_frame(frame, spin, t, wheel):
    """Draw rim circle, predicted zone arc (rotor-referenced), ball point
    and text on a copy of ``frame``."""
    out = frame.copy()
    cal = spin.get("calibration") or {}
    cx, cy, R = cal.get("cx"), cal.get("cy"), cal.get("R")
    if not cx:
        return out
    cv2.circle(out, (int(cx), int(cy)), int(R), (60, 160, 200), 2)
    ser = spin.get("series") or {}
    preds = spin.get("predictions") or []
    pred = preds[-1] if preds else None
    phi0 = _interp(ser.get("t_rotor"), ser.get("phi_rotor"), t)
    step = 360.0 / wheel.n_pockets
    if pred is not None and phi0 is not None:
        # zone arc on the pocket ring (rotor-referenced at this instant)
        zone = pred["zone_adaptive"]["zone"] if pred.get("zone_adaptive") \
            else pred["zone"]
        for z in zone:
            if z not in wheel.pocket_order:
                continue
            idx = wheel.pocket_order.index(z)
            az = math.radians((phi0 + idx * step) % 360.0)
            p0 = (int(cx + 0.50 * R * math.cos(az - math.radians(step / 2))),
                  int(cy + 0.50 * R * math.sin(az - math.radians(step / 2))))
            p1 = (int(cx + 0.50 * R * math.cos(az + math.radians(step / 2))),
                  int(cy + 0.50 * R * math.sin(az + math.radians(step / 2))))
            cv2.line(out, p0, p1, (40, 200, 230), 6)
        # predicted impact azimuth tick
        ti = pred.get("theta_impact_deg")
        if ti is not None:
            az = math.radians(ti)
            p0 = (int(cx + 0.86 * R * math.cos(az)),
                  int(cy + 0.86 * R * math.sin(az)))
            p1 = (int(cx + 1.0 * R * math.cos(az)),
                  int(cy + 1.0 * R * math.sin(az)))
            cv2.line(out, p0, p1, (60, 60, 230), 3)
    # tracked ball point
    th = _interp(ser.get("t_ball"), ser.get("theta_ball"), t)
    if th is not None:
        az = math.radians(th % 360.0)
        p = (int(cx + 0.93 * R * math.cos(az)),
             int(cy + 0.93 * R * math.sin(az)))
        cv2.circle(out, p, 7, (50, 60, 235), -1)
        cv2.circle(out, p, 10, (230, 230, 230), 1)
    # text block
    lines = [f"spin {spin.get('index', '?')}  t={t:.2f}s"]
    if pred is not None:
        lead = pred["t_drop_est"] - pred["t_emit"]
        lines.append(f"zone{len(pred['zone'])}: {pred['zone']}")
        lines.append(f"top {pred['top_pocket']}  "
                     f"chute est. dans {lead:.1f}s")
    y = out.shape[0] - 20 - 22 * len(lines)
    for ln in lines:
        y += 22
        cv2.putText(out, ln, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, ln, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (240, 225, 195), 1, cv2.LINE_AA)
    return out


def _grab_frame(video, fps, t):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# ---------------------------------------------------------------------------
# chronology / feasibility blocks
# ---------------------------------------------------------------------------
def _chronology_table(spin):
    rows = []
    # countdowns of THIS spin only (before the launch; later reads belong
    # to the next spin)
    cds = [e for e in spin["events"] if e["kind"] == "countdown"
           and e["t"] < spin["t_plunge_start"]]
    if cds:
        rows.append(("fin compte à rebours", f"{cds[-1]['t']:.1f} s",
                     f"(dernier lu : {cds[-1]['detail'].get('seconds_left')})"))
    bc = [e for e in spin["events"] if e["kind"] == "bets_close"]
    if bc:
        rows.append(("fermeture des paris", f"{bc[0]['t']:.1f} s",
                     "« No more bets »"))
    rows.append(("lancement (plan plongée)",
                 f"{spin['t_plunge_start']:.1f} s", ""))
    if spin.get("drop"):
        rows.append(("chute bille (mesurée)", f"{spin['drop']['t']:.1f} s",
                     f"ω fin = {spin['drop'].get('omega_end')} °/s"))
    if spin.get("result") is not None:
        rows.append(("résultat OCR", f"{spin['result']}", ""))
    trs = "".join(f"<tr><th>{_esc(a)}</th><td>{_esc(b)}</td>"
                  f"<td class='dim'>{_esc(c)}</td></tr>"
                  for a, b, c in rows)
    return f"<table>{trs}</table>"


def _feasibility_banner(spin):
    gap = (spin.get("feasibility") or {}).get("launch_minus_bets_close_s")
    leads = [p.get("lead_actual_s") for p in spin.get("predictions", [])
             if p.get("lead_actual_s") is not None]
    lead_txt = (f"avance réelle prédiction → chute : "
                f"<b>{max(leads):.1f} s</b> (première émission), "
                f"{leads[-1]:.1f} s (dernière)" if leads else
                "pas de prédiction émise")
    if gap is None:
        return (f'<div class="banner warn">chronologie paris/lancement '
                f'incomplète sur ce spin. {lead_txt}.</div>')
    if gap > 0:
        return (f'<div class="banner warn"><b>Verdict faisabilité : la '
                f'bille est lancée {gap:.1f} s APRÈS la fermeture des '
                f'paris</b> — la prédiction n\'est pas jouable en mise sur '
                f'ce flux. {lead_txt}.</div>')
    return (f'<div class="banner ok"><b>paris encore ouverts au '
            f'lancement</b> ({-gap:.1f} s de marge). {lead_txt}.</div>')


def _verdict_line(spin):
    parts = []
    if spin.get("result") is None:
        return '<span class="dim">pas de résultat OCR associé</span>'
    res = spin["result"]
    preds = spin.get("predictions") or []
    if not preds:
        return f'résultat <b>{res}</b> — aucune prédiction émise'
    last = preds[-1]
    for label, key, extra in (
            ("zone 9 (scatter)", "hit", f"p={last['zone_p']:.2f}"),
            ("zone adaptive", "hit_adaptive",
             f"{last['zone_adaptive']['width']} poches @ "
             f"{last['zone_adaptive']['zone_p']:.2f}"),
            ("zone brute 9 (impact)", "hit_raw", "avant scatter")):
        v = spin.get(key)
        if v is None:
            continue
        cls = "hit" if v else "miss"
        txt = "HIT" if v else "MISS"
        parts.append(f'<span class="{cls}">{txt}</span> '
                     f'<span class="dim">{label}, {extra}</span>')
    off = spin.get("offset_raw")
    off_txt = f" · offset brut (résultat − impact) = {off} poches" \
        if off is not None else ""
    lead = spin.get("lead_s")
    lead_txt = f" · avance dernière prédiction = {lead:.1f} s" \
        if lead is not None else ""
    return (f"résultat <b>{res}</b> · " + " · ".join(parts)
            + f"<br><span class='dim'>{off_txt[3:]}{lead_txt}</span>")


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def render_spin_report(report, video: str, out_html: str,
                       frames_dir: str) -> None:
    """Render a self-contained HTML session report (SPEC.md section G)."""
    os.makedirs(frames_dir, exist_ok=True)
    wheel = WheelConfig()
    try:
        fps = FrameSource(video).fps
    except (FileNotFoundError, IOError):
        fps = 60.0          # fabricated reports / missing video: no grabs
        video = None
    out_dir = os.path.dirname(os.path.abspath(out_html)) or "."
    summ = report.summary()

    body = ["<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>ProphetVision — rapport de session</title>"
            f"<style>{CSS}</style></head><body><div class='wrap'>"]
    body.append("<h1>ProphetVision — rapport de session</h1>")
    body.append(f"<div class='sub'>"
                f"{_esc(os.path.basename(video) if video else '—')} · "
                f"{report.n_frames} frames uniques · moteur "
                f"{report.pipeline_fps:.0f} fps · prédiction = distribution "
                f"de probabilité, pas une certitude</div>")

    # --- session header
    fz = summ["feasibility"]
    cards = [
        ("spins", summ["n_spins"]),
        ("résultats lus", len(summ["results"])),
        ("hits zone 9", f"{summ['hits']}/{len(summ['results'])}"),
        ("hits zone adapt.", f"{summ['hits_adaptive_zone']}/"
         f"{len(summ['results'])}"),
        ("avance max", f"{summ['max_prediction_lead_s']:.1f} s"
         if summ["max_prediction_lead_s"] is not None else "—"),
        ("ω chute appris", f"{report.omega_drop_final} °/s"
         if report.omega_drop_final else "—"),
    ]
    body.append("<div class='cards'>" + "".join(
        f"<div class='card'><div class='v'>{_esc(v)}</div>"
        f"<div class='k'>{_esc(k)}</div></div>" for k, v in cards)
        + "</div>")
    gaps = fz["launch_minus_bets_close_s"]
    if gaps:
        body.append(f"<div class='banner warn'><b>Verdict global :</b> "
                    f"{_esc(fz['verdict'])}</div>")
    sc_svg = scatter_svg(report)
    if sc_svg:
        body.append("<h3>Auto-apprentissage (scatter de rebond)</h3>"
                    f"<div class='panel'>{sc_svg}</div>")
    import json as _json
    body.append("<details class='panel'><summary class='dim'>résumé "
                "session (JSON)</summary><pre style='color:" + _DIM +
                ";font-size:11px;white-space:pre-wrap'>"
                + _esc(_json.dumps(summ, indent=2, ensure_ascii=False))
                + "</pre></details>")

    # --- per spin
    for spin in report.spins:
        i = spin.get("index", "?")
        body.append(f"<h2>Spin {i} — plongée à "
                    f"{spin['t_plunge_start']:.1f} s</h2>")
        body.append(_feasibility_banner(spin))
        preds = spin.get("predictions") or []
        last = preds[-1] if preds else None
        stats = spin.get("track_stats") or {}
        body.append("<div class='grid'>")
        if last is not None:
            body.append("<div class='panel'><h3>roue — probabilités par "
                        "poche (dernière prédiction)</h3>"
                        + wheel_svg(wheel, last.get("probs"), last["zone"],
                                    last.get("zone_adaptive"),
                                    last["top_pocket"], spin.get("result"))
                        + f"<div class='dim'>arc or = zone {len(last['zone'])} "
                          f"(p={last['zone_p']:.2f}) · arc turquoise = zone "
                          f"adaptive ({last['zone_adaptive']['width']} poches "
                          f"@ {last['zone_adaptive']['zone_p']:.2f}) · "
                          f"▲ top poche · ○ rouge = résultat réel</div></div>")
        else:
            body.append("<div class='panel'><h3>roue</h3><p class='dim'>"
                        "aucune prédiction émise sur ce spin</p></div>")
        body.append("<div class='panel'><h3>vitesses angulaires</h3>"
                    + omega_svg(spin)
                    + f"<div class='dim'>arc suivi : "
                      f"{stats.get('arc_deg', 0):.0f} ° · v₀ ≈ "
                      f"{stats.get('v0') or '—'} °/s · couverture "
                      f"{(stats.get('coverage') or 0) * 100:.0f} %</div>"
                      "</div>")
        body.append("</div><div class='grid'>")
        body.append("<div class='panel'><h3>chronologie</h3>"
                    + _chronology_table(spin))
        ev_rows = "".join(
            f"<tr><td>{e['t']:.2f}</td><td>{_esc(e['kind'])}</td>"
            f"<td class='dim'>{_esc(', '.join(f'{k}={v}' for k, v in e.get('detail', {}).items()))}</td></tr>"
            for e in spin.get("events", []))
        body.append("<h3>événements bruts</h3><table><tr><th>t (s)</th>"
                    f"<th>type</th><th>détail</th></tr>{ev_rows}</table>"
                    "</div>")
        body.append("<div class='panel'><h3>verdict</h3><p>"
                    + _verdict_line(spin) + "</p>")
        if preds:
            rows = "".join(
                f"<tr><td>{p['t_emit']:.2f}</td><td>{p['t_drop_est']:.2f}"
                f"</td><td>{p['t_drop_est'] - p['t_emit']:.2f}</td>"
                f"<td>{(f'{p['lead_actual_s']:.2f}' if p.get('lead_actual_s') is not None else '—')}</td>"
                f"<td>{p['top_pocket']}</td><td>{p['zone_p']:.2f}</td></tr>"
                for p in preds)
            body.append("<h3>prédictions (rafraîchies ~2×/s, la dernière "
                        "remplace la précédente)</h3><table><tr><th>t "
                        "émission</th><th>t chute est.</th><th>avance est."
                        " (s)</th><th>avance réelle (s)</th><th>top</th>"
                        "<th>p zone</th></tr>" + rows + "</table>")
        body.append("</div></div>")

        # annotated frames
        shots = []
        if preds:
            shots.append((preds[0]["t_emit"], "pred"))
        if spin.get("drop"):
            shots.append((spin["drop"]["t"], "drop"))
        imgs = []
        for t, tag in shots[:2]:
            if video is None:
                break
            frame = _grab_frame(video, fps, t)
            if frame is None:
                continue
            ann = annotate_frame(frame, spin, t, wheel)
            name = f"spin{i}_{tag}.jpg"
            path = os.path.join(frames_dir, name)
            cv2.imwrite(path, ann, [cv2.IMWRITE_JPEG_QUALITY, 88])
            rel = os.path.relpath(path, out_dir)
            imgs.append(f"<img class='frame' src='{_esc(rel)}' "
                        f"alt='frame annotée {tag} t={t:.1f}s'>")
        if imgs:
            body.append("<div class='panel'><h3>frames annotées</h3>"
                        + "".join(imgs) + "</div>")

    body.append("<div class='legal'>ProphetVision — outil de recherche. "
                "La prédiction est une distribution de probabilité, pas une "
                "certitude. Respectez les conditions d'utilisation des "
                "opérateurs et la législation en vigueur.</div>")
    body.append("</div></body></html>")
    with open(out_html, "w") as f:
        f.write("".join(body))
