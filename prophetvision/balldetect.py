"""Learned ball detector (ONNX YOLO) with tiled native-resolution inference.

Classical CV never solved the ball in the oblique view (see README). A trained
detector is the natural next tool, and this wraps the ONNX model shipped with
the repository.

Two things matter for using it correctly on this footage, and both were found
by measurement rather than assumption:

* **Scale.** The network takes 320x320. Fed a whole 1206x910 frame, the ball
  shrinks to ~4 px and is never found. Run on a native-resolution 320 crop
  centred on the ball it localises it to ~5 px at 0.8 confidence. So inference
  must be *tiled at native resolution*, never whole-frame.

* **The turret is a hard negative.** On a top-down window where the ball is
  certainly on the rim, a naive tiled sweep returned 417 detections at radius
  0.4-0.6 R (the gold hub and its arms — round, bright, ball-coloured) and only
  3 on the rim band where the ball actually was. Any use of this detector needs
  the radial gate and the confidence floor applied here, and callers should
  still treat a bare detection stream as unverified: run
  :func:`~prophetvision.vmf.detector_power` on a window with a known ball
  before trusting a negative result.

The class deliberately reports its own hit rate against a reference track
(:meth:`evaluate`) so its usable operating range is measured, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # optional dependency
    import onnxruntime as ort
except Exception:  # pragma: no cover - exercised only without onnxruntime
    ort = None

import cv2

DEFAULT_MODEL = "ball_sota_yolo11n_320_fp16.onnx"


@dataclass
class Detection:
    t: float
    x: float
    y: float
    confidence: float


class BallDetector:
    """Tiled ONNX ball detector.

    ``r_gate`` restricts accepted detections to a radial band of the wheel (in
    units of the wheel radius / ellipse parameter), which is what suppresses
    the turret. ``border`` drops detections hugging a tile edge, where the
    letterbox padding produces spurious boxes; the tile stride must therefore
    be small enough that every pixel is interior to at least one tile.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL, tile: int = 320,
                 stride: int = 120, border: int = 26,
                 conf_threshold: float = 0.45, nms_radius: float = 18.0,
                 threads: int = 4):
        # Validate the tiling geometry before loading anything: a stride wider
        # than a tile's interior leaves pixels that every tile sees only inside
        # its rejected border, so the ball would be silently unreachable there.
        if stride > tile - 2 * border:
            raise ValueError(
                f"stride {stride} leaves a band no tile sees as interior; "
                f"use stride <= {tile - 2 * border}")
        if ort is None:
            raise RuntimeError("onnxruntime is required for BallDetector")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            model_path, sess_options=opts,
            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.tile = tile
        self.stride = stride
        self.border = border
        self.conf_threshold = conf_threshold
        self.nms_radius = nms_radius

    # ------------------------------------------------------------------
    def _infer_tile(self, tile_bgr: np.ndarray) -> list[tuple[float, float, float]]:
        x = tile_bgr[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        y = self.session.run(None, {self.input_name: x})[0][0]  # (5, N)
        out = []
        b = self.border
        for j in np.nonzero(y[4] > self.conf_threshold)[0]:
            cx, cy = float(y[0, j]), float(y[1, j])
            if cx < b or cx > self.tile - b or cy < b or cy > self.tile - b:
                continue
            out.append((cx, cy, float(y[4, j])))
        return out

    def detect_frame(self, frame: np.ndarray, t: float = 0.0,
                     roi: tuple[int, int, int, int] | None = None,
                     accept=None) -> list[Detection]:
        """Detections in one frame. ``accept(x, y) -> bool`` is the radial gate."""
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = roi or (0, 0, w, h)
        xs = list(range(x0, max(x0 + 1, x1 - self.tile + 1), self.stride))
        ys = list(range(y0, max(y0 + 1, y1 - self.tile + 1), self.stride))
        if x1 - self.tile not in xs:
            xs.append(max(x0, x1 - self.tile))
        if y1 - self.tile not in ys:
            ys.append(max(y0, y1 - self.tile))
        cands: list[tuple[float, float, float]] = []
        for yy in ys:
            for xx in xs:
                tile = frame[yy:yy + self.tile, xx:xx + self.tile]
                if tile.shape[0] != self.tile or tile.shape[1] != self.tile:
                    continue
                for cx, cy, conf in self._infer_tile(tile):
                    X, Y = cx + xx, cy + yy
                    if accept is not None and not accept(X, Y):
                        continue
                    cands.append((X, Y, conf))
        cands.sort(key=lambda c: -c[2])
        kept: list[tuple[float, float, float]] = []
        for c in cands:
            if all(np.hypot(c[0] - k[0], c[1] - k[1]) > self.nms_radius
                   for k in kept):
                kept.append(c)
        return [Detection(t=t, x=c[0], y=c[1], confidence=c[2]) for c in kept]

    # ------------------------------------------------------------------
    @staticmethod
    def evaluate(detections: list[Detection], reference,
                 tol_px: float = 25.0) -> dict:
        """Hit rate of a detection stream against a reference ball track.

        ``reference`` is an iterable of (t, x, y). Reports the fraction of
        reference instants with a detection within ``tol_px`` — the number that
        decides whether a negative result from this detector means anything.
        """
        ref = list(reference)
        if not ref:
            return {"instants": 0, "hits": 0, "hit_rate": 0.0,
                    "median_error_px": float("nan")}
        by_t: dict[float, list[Detection]] = {}
        for d in detections:
            by_t.setdefault(round(d.t, 3), []).append(d)
        times = np.array(sorted(by_t)) if by_t else np.zeros(0)
        hits, errors = 0, []
        for t, x, y in ref:
            if len(times) == 0:
                continue
            nearest = times[np.argmin(np.abs(times - t))]
            if abs(nearest - t) > 0.03:
                continue
            dists = [np.hypot(d.x - x, d.y - y) for d in by_t[nearest]]
            if dists:
                errors.append(min(dists))
                hits += min(dists) <= tol_px
        return {
            "instants": len(ref),
            "hits": int(hits),
            "hit_rate": hits / len(ref),
            "median_error_px": float(np.median(errors)) if errors else float("nan"),
        }


def radial_gate(cx: float, cy: float, ax: float, ay: float,
                r_lo: float, r_hi: float):
    """Accept-callback keeping detections inside an elliptical annulus."""
    def accept(x: float, y: float) -> bool:
        r = float(np.hypot((x - cx) / ax, (y - cy) / ay))
        return r_lo <= r <= r_hi
    return accept
