"""Game chronology: countdown OCR, bets close, result, banner. SPEC.md section D.

All detection is done per-frame (streaming) on the 1206x910 Gravity Auto
Roulette layout:

- countdown : green/yellow/red circular timer on the RIGHT edge of the screen
  (digit at approx x 1097-1123, y 799-833). OCR by template matching.
- bets_close : white bold "No more bets" text centered in the upper band
  (y 150-250). Detected by bright-pixel energy with temporal persistence.
- result : bright red rounded marker above the winning pocket with a white
  number inside (plunge view). OCR of the digits inside the marker.
- banner : history strip at the very top (y 15-65), ~16 numbers, most recent
  on the LEFT, white or red digits on dark pastilles. Full OCR.

Digit OCR uses normalized cross-correlation against templates cropped from the
real videos (prophetvision/data/digit_templates.npz). No external OCR
dependency. If the npz is missing, synthetic bold sans-serif templates are
rendered with cv2.putText as a fallback.
"""
from __future__ import annotations

import io
import os
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Screen layout, expressed on 1206x910 reference frames. All ROIs and pixel
# thresholds are scaled to the actual frame size at runtime (see
# Chronology._scale), so the detector works at any resolution with the
# same screen layout (e.g. 954x720).
# ---------------------------------------------------------------------------
REF_W, REF_H = 1206.0, 910.0
BANNER_Y0, BANNER_Y1 = 15, 65
COUNTDOWN_X0, COUNTDOWN_Y0 = 1080, 785
COUNTDOWN_X1, COUNTDOWN_Y1 = 1140, 850
BETS_X0, BETS_Y0, BETS_X1, BETS_Y1 = 400, 150, 800, 250

_TEMPLATE_FILE = os.path.join(os.path.dirname(__file__), "data",
                              "digit_templates.npz")
_GLYPH_W, _GLYPH_H = 24, 36  # canonical glyph size for matching


@dataclass
class SpinEvent:
    kind: str
    t: float
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Digit template OCR
# ---------------------------------------------------------------------------
def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two flattened glyph images."""
    av = a.ravel().astype(np.float64)
    bv = b.ravel().astype(np.float64)
    av -= av.mean()
    bv -= bv.mean()
    d = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(av @ bv / d) if d > 1e-12 else 0.0


class DigitOCR:
    """Nearest-template digit classifier (normalized cross-correlation)."""

    def __init__(self, templates_path: str | None = None):
        path = templates_path or _TEMPLATE_FILE
        if os.path.exists(path):
            z = np.load(path)
            self._X = z["X"].astype(np.float32)
            self._y = z["y"].astype(np.int64)
        else:
            try:  # embedded templates (npz base64-encoded in a .py module)
                from .data.digit_templates_b64 import npz_bytes
                z = np.load(io.BytesIO(npz_bytes()))
                self._X = z["X"].astype(np.float32)
                self._y = z["y"].astype(np.int64)
            except Exception:  # fallback: synthetic bold sans-serif glyphs
                self._X, self._y = self._synthetic_templates()

    @staticmethod
    def _synthetic_templates() -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for d in range(10):
            canvas = np.zeros((_GLYPH_H + 12, _GLYPH_W + 12), np.uint8)
            cv2.putText(canvas, str(d), (3, _GLYPH_H + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, 255, 3, cv2.LINE_AA)
            X.append(_normalize_glyph(canvas))
            y.append(d)
        return np.stack(X).astype(np.float32), np.asarray(y, np.int64)

    def classify(self, glyph_mask: np.ndarray) -> tuple[int, float]:
        """(digit, score) for a binary glyph mask (any size)."""
        q = _normalize_glyph(glyph_mask)
        scores = np.array([_corr(q, tp) for tp in self._X])
        k = int(np.argmax(scores))
        return int(self._y[k]), float(scores[k])


def _normalize_glyph(mask: np.ndarray) -> np.ndarray:
    """Crop to bounding box of lit pixels and resize to canonical size."""
    m = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return np.zeros((_GLYPH_H, _GLYPH_W), np.float32)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(m.astype(np.float32), (_GLYPH_W, _GLYPH_H),
                      interpolation=cv2.INTER_AREA)


def _digit_binary(frame_i16: np.ndarray) -> np.ndarray:
    """Binary mask of digit-like pixels: bright white OR saturated red."""
    r = frame_i16[..., 2]
    g = frame_i16[..., 1]
    b = frame_i16[..., 0]
    white = frame_i16.min(axis=2) > 110
    red = (r > 110) & (r - g > 40) & (r - b > 40)
    return (white | red).astype(np.uint8)


# ---------------------------------------------------------------------------
# Chronology
# ---------------------------------------------------------------------------
class Chronology:
    """Streaming detector of game-phase events (SPEC D).

    Usage: call ``process(t, frame_bgr)`` for each frame in time order;
    it returns the list of SpinEvents detected at that instant (usually
    empty). Internal state de-duplicates events within a spin.
    """

    def __init__(self, templates_path: str | None = None,
                 banner_period: float = 0.5):
        self.ocr = DigitOCR(templates_path)
        self.banner_period = float(banner_period)
        # countdown state
        self._cd_last_value: int | None = None
        self._cd_last_t: float = -1e9
        # bets_close state
        self._bets_high_since: float | None = None
        self._bets_low_since: float | None = None
        self._bets_armed = True
        # result state
        self._frame_idx = -1
        self._marker_votes: deque = deque(maxlen=240)  # (t, number, x, w)
        self._result_armed = True
        self._marker_absent_since: float | None = None
        self._last_result_number: int | None = None
        # banner state
        self._banner_last_check: float = -1e9
        self._banner_pending: list[int] | None = None
        self._banner_emitted: list[int] | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def _scale(frame: np.ndarray) -> tuple[float, float]:
        """(sx, sy): frame size relative to the 1206x910 reference."""
        h, w = frame.shape[:2]
        return w / REF_W, h / REF_H

    def process(self, t: float, frame: np.ndarray) -> list[SpinEvent]:
        events: list[SpinEvent] = []
        ev = self._countdown(t, frame)
        if ev is not None:
            events.append(ev)
        ev = self._bets_close(t, frame)
        if ev is not None:
            events.append(ev)
        # The result marker persists ~4 s; checking every 3rd frame keeps
        # the per-frame cost low without risking a miss.
        self._frame_idx += 1
        if self._frame_idx % 3 == 0 or self._marker_votes:
            ev = self._result(t, frame)
            if ev is not None:
                events.append(ev)
        ev = self._banner(t, frame)
        if ev is not None:
            events.append(ev)
        return events

    # ------------------------------------------------------------------
    def _countdown(self, t: float, frame: np.ndarray) -> SpinEvent | None:
        sx, sy = self._scale(frame)
        box = frame[int(COUNTDOWN_Y0 * sy):int(COUNTDOWN_Y1 * sy),
                    int(COUNTDOWN_X0 * sx):int(COUNTDOWN_X1 * sx)
                    ].astype(np.int16)
        if box.size == 0:
            return None
        mn = box.min(axis=2)
        # Adaptive threshold: the digit is much brighter than the dark
        # background but can be dim gray on some frames.
        thr = max(80.0, float(np.percentile(mn, 90)) * 0.7)
        m = (mn > thr).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        best = None
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a > 40 * sx * sy and h > 15 * sy:
                if best is None or a > stats[best, 4]:
                    best = i
        if best is None:
            return None
        glyph = (lab == best).astype(np.uint8)
        digit, score = self.ocr.classify(glyph)
        if score < 0.70:
            return None
        # Anti-duplicate: emit only when the value changes (or after a long
        # silence, i.e. a new spin).
        if (digit != self._cd_last_value) or (t - self._cd_last_t > 20.0):
            self._cd_last_value = digit
            self._cd_last_t = t
            return SpinEvent("countdown", t, {"seconds_left": digit,
                                              "score": round(score, 3)})
        self._cd_last_t = t
        return None

    # ------------------------------------------------------------------
    def _bets_close(self, t: float, frame: np.ndarray) -> SpinEvent | None:
        sx, sy = self._scale(frame)
        band = frame[int(BETS_Y0 * sy):int(BETS_Y1 * sy),
                     int(BETS_X0 * sx):int(BETS_X1 * sx)].astype(np.int16)
        energy = int((band.min(axis=2) > 140).sum())
        e_scale = sx * sy  # energy is a pixel count
        event = None
        if energy > 1500 * e_scale:
            if self._bets_high_since is None:
                self._bets_high_since = t
            elif (self._bets_armed
                    and t - self._bets_high_since >= 0.3):
                event = SpinEvent("bets_close", self._bets_high_since,
                                  {"energy": energy})
                self._bets_armed = False
            self._bets_low_since = None
        elif energy < 600 * e_scale:
            if self._bets_low_since is None:
                self._bets_low_since = t
            elif t - self._bets_low_since > 1.0:
                self._bets_armed = True
                self._bets_high_since = None
        return event

    # ------------------------------------------------------------------
    def _find_marker(self, frame: np.ndarray):
        """Bright red rounded rectangle of the result overlay, if present.

        Detection runs at half resolution (the marker is a large solid
        blob); the returned bbox is rescaled to full resolution.
        """
        sx, sy = self._scale(frame)
        small = cv2.resize(frame, None, fx=0.5, fy=0.5)
        f = small.astype(np.int16)
        r, g, b = f[..., 2], f[..., 1], f[..., 0]
        m = ((r > 110) & (r - g > 45) & (r - b > 45)).astype(np.uint8)
        m[:int(35 * sy), :] = 0  # ignore the history banner
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        best = None
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if (a < 175 * sx * sy or not (10 * sx <= w <= 45 * sx)
                    or not (13 * sy <= h <= 45 * sy)):
                continue
            if not (0.35 <= w / h <= 1.8):
                continue
            if a / (w * h) < 0.45:  # solidity: marker is a solid rectangle
                continue
            if best is None or a > stats[best, 4]:
                best = i
        if best is None:
            return None, None
        x, y, w, h, a = stats[best]
        return None, (int(2 * x), int(2 * y), int(2 * w), int(2 * h))

    def _read_marker_number(self, frame: np.ndarray,
                            bbox) -> tuple[int, float] | None:
        sx, sy = self._scale(frame)
        x, y, w, h = bbox
        crop = frame[y:y + h, x:x + w].astype(np.int16)
        m = (crop.min(axis=2) > 110).astype(np.uint8)  # white digits on red
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        # Candidate digit glyphs: plausible size, then classified; only
        # confident reads are kept (rejects the ball circle and the marker
        # border highlight, which appear as extra components at 720p).
        cands = []
        for i in range(1, n):
            xx, yy, ww, hh, aa = stats[i]
            if aa > 25 * sx * sy and 12 * sy < hh <= 0.65 * h:
                d, s = self.ocr.classify((lab == i).astype(np.uint8))
                if s >= 0.60:
                    cands.append((xx, d, s))
        if not 1 <= len(cands) <= 2:
            return None
        cands.sort()
        digits = [d for _, d, _ in cands]
        scores = [s for _, _, s in cands]
        number = int("".join(map(str, digits)))
        if not (0 <= number <= 36):
            return None
        return number, float(min(scores))

    def _result(self, t: float, frame: np.ndarray) -> SpinEvent | None:
        mask, bbox = self._find_marker(frame)
        if bbox is None:
            if self._marker_absent_since is None:
                self._marker_absent_since = t
            elif t - self._marker_absent_since > 1.5:
                self._result_armed = True
                self._marker_votes.clear()
            return None
        self._marker_absent_since = None
        read = self._read_marker_number(frame, bbox)
        if read is None:
            return None
        number, score = read
        if score < 0.60:
            return None
        x, y, w, h = bbox
        self._marker_votes.append((t, number, x, w))
        # Persistence: the marker fades in over ~0.5 s, during which partial
        # reads (single digit, misread glyph) are common. Only trust votes
        # with a full-width marker, and require a strong consensus: modal
        # number in >= 75 % of the last 0.6 s of votes, the 3 most recent
        # votes agreeing, and a vote span of >= 0.35 s (duplicated frames
        # inflate vote counts, so count alone is not enough).
        sx, _ = self._scale(frame)
        recent = [(tt, nn, xx, ww) for tt, nn, xx, ww in self._marker_votes
                  if t - tt <= 0.6 and ww >= 28 * sx]
        if len(recent) < 4:
            return None
        numbers = [nn for _, nn, _, _ in recent]
        modal = max(set(numbers), key=numbers.count)
        # The modal number must persist by itself for >= 0.5 s: during the
        # ~0.5 s marker fade-in, partial reads (e.g. "25" seen as "2") can
        # dominate a short window, but they never persist once the marker
        # is fully drawn.
        modal_times = [tt for tt, nn, _, _ in recent if nn == modal]
        modal_span = max(modal_times) - min(modal_times)
        xs = [xx for _, _, xx, _ in recent]
        stable = (max(xs) - min(xs) <= 12 * sx)
        consensus = (numbers.count(modal) / len(numbers) >= 0.75
                     and [nn for _, nn, _, _ in recent[-3:]] == [modal] * 3)
        if (self._result_armed and modal_span >= 0.5 and stable
                and consensus):
            self._result_armed = False
            self._last_result_number = modal
            validated = bool(self._banner_emitted
                             and self._banner_emitted[0] == modal)
            return SpinEvent("result", t, {"number": modal,
                                           "score": round(score, 3),
                                           "banner_validated": validated})
        return None

    # ------------------------------------------------------------------
    def read_banner(self, frame: np.ndarray) -> list[int]:
        """OCR of the history strip: list of numbers, most recent first."""
        sx, sy = self._scale(frame)
        strip = frame[int(BANNER_Y0 * sy):int(BANNER_Y1 * sy)].astype(np.int16)
        m = _digit_binary(strip)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        glyphs = []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a > 25 * sx * sy and h > 12 * sy:
                glyphs.append((int(x), int(x + w), i))
        glyphs.sort()
        groups: list[list[int | None]] = []
        cur: list[int | None] = []
        cur_end = -100
        for x0, x1, i in glyphs:
            if x0 - cur_end > 15 * sx:  # new pastille
                if cur:
                    groups.append(cur)
                cur = []
            d, s = self.ocr.classify((lab == i).astype(np.uint8))
            cur.append(d if s >= 0.55 else None)
            cur_end = max(cur_end, x1)
        if cur:
            groups.append(cur)
        numbers = []
        for g in groups:
            if any(d is None for d in g) or not (1 <= len(g) <= 2):
                continue
            v = int("".join(str(d) for d in g))
            if 0 <= v <= 36:
                numbers.append(v)
        return numbers

    def _banner(self, t: float, frame: np.ndarray) -> SpinEvent | None:
        if t - self._banner_last_check < self.banner_period:
            return None
        self._banner_last_check = t
        numbers = self.read_banner(frame)
        if not numbers:
            return None
        # Require two consecutive identical reads (0.5 s apart) to emit,
        # and emit only on change (anti-duplicate).
        if numbers == self._banner_pending and numbers != self._banner_emitted:
            self._banner_emitted = list(numbers)
            return SpinEvent("banner", t, {"numbers": list(numbers)})
        self._banner_pending = numbers
        return None
