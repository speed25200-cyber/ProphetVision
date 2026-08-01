"""Streaming frame source, dedup, shot segmentation. See SPEC.md section A."""
from __future__ import annotations

import os
from typing import Iterator

import cv2
import numpy as np

# Small grayscale size used for inter-frame differences (SPEC: 300x226).
DIFF_SIZE = (300, 226)


def _small_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, DIFF_SIZE, interpolation=cv2.INTER_AREA)


def _frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference (0-255 scale) between two small grayscales."""
    return float(np.mean(cv2.absdiff(a, b)))


class FrameSource:
    """Streaming iterator over a video file: one frame at a time, never the
    whole video in memory (RAM constraint, SPEC).

    Yields ``(frame_index, t_seconds, frame_bgr)`` where ``t_seconds`` is the
    container time since the start of the stream (``frame_index / fps``).
    """

    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"video not found: {path}")
        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"cannot open video: {path}")
        self._fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def duration(self) -> float:
        return self._n_frames / self._fps if self._fps else 0.0

    @property
    def frame_size(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        return (self._width, self._height)

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        # Each iteration opens its own capture, so several independent
        # iterators (e.g. dedup + raw consumption) never interfere.
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise IOError(f"cannot open video: {self.path}")
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, idx / self._fps, frame
                idx += 1
        finally:
            cap.release()


def dedup_times(
    frames_iter: Iterator[tuple[int, float, np.ndarray]],
    diff_thresh: float = 0.08,
) -> Iterator[tuple[float, np.ndarray]]:
    """Filter out duplicated frames, yielding ``(t_seconds, frame)`` only for
    genuinely new frames.

    A frame is a duplicate of the previous container frame when the L1 norm
    (mean absolute difference, 0-255 scale) of their 300x226 grayscales is
    below ``diff_thresh``.

    The emitted time is the container time of the FIRST occurrence of each
    unique frame: a displayed frame occupies its container slot, so its
    container time already is the real time (counting the duplicates).
    """
    prev = None
    for _idx, t, frame in frames_iter:
        small = _small_gray(frame)
        if prev is not None and _frame_diff(small, prev) < diff_thresh:
            continue  # duplicate: drop it
        prev = small
        yield t, frame


class ShotSegmenter:
    """Classifies each frame as 'oblique' | 'plunge' | 'other'.

    Rules (SPEC A):
    - a shot cut is a frame-to-frame L1 diff (300x226 gray, 0-255 scale)
      greater than ``cut_thresh`` (~8);
    - 'plunge' when a Hough circle with R in [250, 420] px centered on the
      image is detected;
    - else 'oblique' when a large flat ellipse (the wheel rim in oblique
      view) is found;
    - else 'other'.

    Hysteresis / anti-flicker policy:
    - a shot only really changes at a CUT: after a cut, the new class must be
      voted ``hysteresis`` consecutive times before the state switches;
    - without a cut, a disagreeing NON-'other' vote must persist for
      ``persist_votes`` consecutive votes to switch (safety net for a missed
      cut);
    - 'other' votes never replace a confident state on their own: an overlay
      occluding the wheel is not a new shot.
    """

    def __init__(
        self,
        cut_thresh: float = 8.0,
        hysteresis: int = 2,
        vote_every: int = 3,
        persist_votes: int = 30,
        circle_r_min: float = 250.0,
        circle_r_max: float = 420.0,
        center_frac: float = 0.12,
    ):
        self.cut_thresh = float(cut_thresh)
        self.hysteresis = int(hysteresis)
        self.vote_every = max(1, int(vote_every))
        self.persist_votes = int(persist_votes)
        self.circle_r_min = float(circle_r_min)
        self.circle_r_max = float(circle_r_max)
        # Max distance of the circle center from the image center, as a
        # fraction of min(w, h), for the circle to count as a plunge shot.
        self.center_frac = float(center_frac)
        self._prev_small: np.ndarray | None = None
        self._state: str | None = None
        self._candidate: str | None = None
        self._candidate_count = 0
        self._frame_count = 0
        self._last_vote: str | None = None
        self._confirming = False

    # ------------------------------------------------------------------ votes
    def _vote_plunge(self, frame: np.ndarray) -> bool:
        """Hough circle R in [r_min, r_max] centered on the image?"""
        h, w = frame.shape[:2]
        scale = 0.5  # half-res Hough: ~4x faster, same geometry/2
        small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=small.shape[1] // 2,
            param1=100,
            param2=40,
            minRadius=int(self.circle_r_min * scale),
            maxRadius=int(self.circle_r_max * scale),
        )
        if circles is None:
            return False
        max_dc = self.center_frac * min(w, h)
        for c in circles[0]:
            cx, cy = float(c[0]) / scale, float(c[1]) / scale
            if np.hypot(cx - w / 2.0, cy - h / 2.0) <= max_dc:
                return True
        return False

    def _vote_oblique(self, frame: np.ndarray) -> bool:
        """Large flat ellipse (wheel rim seen obliquely) present?"""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        band = np.zeros_like(edges)
        band[int(0.30 * h):int(0.72 * h), int(0.20 * w):int(0.80 * w)] = 255
        edges = cv2.bitwise_and(edges, band)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = [c for c in contours if len(c) >= 100]
        contours.sort(key=len, reverse=True)
        for c in contours[:5]:
            try:
                (cx, cy), (d1, d2), _ang = cv2.fitEllipse(c)
            except cv2.error:
                continue
            a, b = max(d1, d2), min(d1, d2)
            if (
                a >= 0.35 * w
                and b / a <= 0.6
                and abs(cx - w / 2.0) < 0.2 * w
                and abs(cy - 0.48 * h) < 0.2 * h
            ):
                return True
        return False

    def _vote(self, frame: np.ndarray) -> str:
        if self._vote_plunge(frame):
            return "plunge"
        if self._vote_oblique(frame):
            return "oblique"
        return "other"

    # ----------------------------------------------------------------- public
    def process(self, frame: np.ndarray,
                small: np.ndarray | None = None) -> str:
        """Classify one frame, with cut detection + hysteresis.

        ``small`` optionally provides the precomputed 300x226 grayscale of
        ``frame`` (e.g. shared with dedup_times in a streaming loop) to
        avoid computing it twice per frame."""
        if small is None:
            small = _small_gray(frame)
        cut = False
        if self._prev_small is not None:
            cut = _frame_diff(small, self._prev_small) > self.cut_thresh
        self._prev_small = small

        if self._state is None:
            # First frame: classify immediately.
            self._state = self._vote(frame)
            self._last_vote = self._state
            self._frame_count += 1
            return self._state

        if cut:
            # Real (or spurious-overlay) cut: re-classify, vote every frame
            # until the new class is confirmed `hysteresis` times in a row.
            self._confirming = True
            self._candidate = None
            self._candidate_count = 0

        want_vote = (
            self._confirming or self._frame_count % self.vote_every == 0
        )
        if want_vote:
            vote = self._vote(frame)
            self._last_vote = vote
            if self._confirming:
                if vote == self._state:
                    # Same class as before the cut: spurious cut (overlay).
                    self._candidate_count += 1
                    if self._candidate_count >= self.hysteresis:
                        self._confirming = False
                        self._candidate = None
                        self._candidate_count = 0
                elif vote == self._candidate:
                    self._candidate_count += 1
                    if self._candidate_count >= self.hysteresis:
                        self._state = vote
                        self._confirming = False
                        self._candidate = None
                        self._candidate_count = 0
                else:
                    self._candidate = vote
                    self._candidate_count = 1
            else:
                if vote == self._state or vote == "other":
                    # 'other' alone never flips a confident state (occlusion
                    # by an overlay is not a new shot).
                    self._candidate = None
                    self._candidate_count = 0
                elif vote == self._candidate:
                    self._candidate_count += 1
                    if self._candidate_count >= self.persist_votes:
                        # Missed-cut safety net: persistent disagreement.
                        self._state = vote
                        self._candidate = None
                        self._candidate_count = 0
                else:
                    self._candidate = vote
                    self._candidate_count = 1
        self._frame_count += 1
        return self._state
