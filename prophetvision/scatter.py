"""Scatter model: what happens between "ball leaves the rim" and "ball rests
in a pocket" is chaotic (deflector hits, bounces). It is treated
probabilistically: a circular distribution over the pocket offset between the
predicted impact pocket and the final resting pocket, learned online from
observed spins and smoothed with a wrapped-Gaussian kernel."""

from __future__ import annotations

import json

import numpy as np


class ScatterModel:
    def __init__(self, n_pockets: int = 37, kernel_sigma_pockets: float = 2.0,
                 prior_weight: float = 1.0):
        self.n = n_pockets
        self.sigma = kernel_sigma_pockets
        # Dirichlet-style uniform prior so the model degrades gracefully to
        # "uniform over the wheel" when it has seen nothing.
        self.counts = np.full(n_pockets, prior_weight / n_pockets, dtype=float)

    # ------------------------------------------------------------------
    def observe(self, offset_pockets: int) -> None:
        """Record one observed offset (final pocket index - impact pocket
        index, modulo n, in the direction of ball travel)."""
        self.counts[offset_pockets % self.n] += 1.0

    def distribution(self) -> np.ndarray:
        """Smoothed probability over offsets (length n, sums to 1)."""
        k = np.arange(self.n)
        d = np.minimum(k, self.n - k).astype(float)
        kernel = np.exp(-0.5 * (d / self.sigma) ** 2)
        kernel /= kernel.sum()
        sm = np.real(np.fft.ifft(np.fft.fft(self.counts) * np.fft.fft(kernel)))
        sm = np.clip(sm, 1e-12, None)
        return sm / sm.sum()

    def n_observations(self) -> float:
        return float(self.counts.sum())

    # ------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps({"n": self.n, "sigma": self.sigma,
                           "counts": self.counts.tolist()})

    @classmethod
    def from_json(cls, s: str) -> "ScatterModel":
        d = json.loads(s)
        m = cls(n_pockets=d["n"], kernel_sigma_pockets=d["sigma"],
                prior_weight=0.0)
        m.counts = np.asarray(d["counts"], dtype=float)
        return m
