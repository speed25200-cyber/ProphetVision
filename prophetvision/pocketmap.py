"""Naming the pocket: from a predicted azimuth to a pocket number.

The physics predicts *where* the ball meets the rotor — an azimuth relative to
the rotor's reference mark. Turning that into a pocket **number** needs one
extra constant: the angular offset between the mark the vision system locks
onto and the pocket the wheel's own numbering starts from. That offset depends
on the wheel and the camera, not on the spin, so it is measured once.

Two ways of measuring it were tried on the reference footage and both failed,
which is why this module asks for a labelled spin instead:

* **the settled ball** — at rest it is small and dim, and at the pocket-ring
  radius it is surrounded by bright white separators and numerals that the
  detector fires on just as readily; filtering hard enough on the cream colour
  to exclude them left 1-2 usable frames out of hundreds;
* **the result marker** — the red card the game draws over the winning number
  looks like a physical marker but is a screen overlay: measured across
  95-98.6 s its azimuth stayed at 269.4 +- 0.5 deg while the wheel turned
  through 120 deg. It is fixed to the screen, not to the pocket.

So: run one spin whose outcome is known (read it off the game's own history
strip), call :func:`calibrate_offset`, and every later spin on that wheel and
camera is named with it. :class:`PocketMap` also reports how many labelled
spins agree, because a single sample cannot reveal a mis-measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import WheelConfig


def _circular_mean_index(indices, n: int) -> float:
    ang = np.radians(np.asarray(indices, dtype=float) * (360.0 / n))
    m = np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))
    return float((m % 360.0) / (360.0 / n))


def calibrate_offset(measured_index: float, true_pocket: int,
                     wheel: WheelConfig | None = None) -> float:
    """Offset (in pockets) taking a measured index to the true pocket index."""
    wheel = wheel or WheelConfig()
    return float((wheel.index_of(true_pocket) - measured_index)
                 % wheel.n_pockets)


@dataclass
class PocketMap:
    """Measured index -> pocket number, calibrated from labelled spins."""

    wheel: WheelConfig = field(default_factory=WheelConfig)
    offset: float | None = None
    samples: list[tuple[float, int]] = field(default_factory=list)

    def observe(self, measured_index: float, true_pocket: int) -> None:
        """Record a labelled spin and refresh the offset."""
        self.samples.append((float(measured_index), int(true_pocket)))
        offs = [calibrate_offset(m, p, self.wheel) for m, p in self.samples]
        self.offset = _circular_mean_index(offs, self.wheel.n_pockets)

    @property
    def spread_pockets(self) -> float:
        """Circular spread of the per-spin offsets, in pockets.

        Above ~1 pocket the calibration is not trustworthy: some measurement is
        picking a different feature on different spins.
        """
        if len(self.samples) < 2:
            return float("nan")
        n = self.wheel.n_pockets
        offs = np.array([calibrate_offset(m, p, self.wheel)
                         for m, p in self.samples])
        ang = np.radians(offs * (360.0 / n))
        r = abs(np.exp(1j * ang).mean())
        r = min(max(r, 1e-12), 1.0 - 1e-12)
        return float(np.degrees(np.sqrt(-2.0 * np.log(r))) / (360.0 / n))

    @property
    def is_trustworthy(self) -> bool:
        return (self.offset is not None and len(self.samples) >= 2
                and self.spread_pockets <= 1.0)

    def pocket(self, measured_index: float) -> int:
        if self.offset is None:
            raise RuntimeError(
                "PocketMap is uncalibrated: observe() at least one spin whose "
                "outcome you know before asking for a pocket number")
        n = self.wheel.n_pockets
        idx = int(round(measured_index + self.offset)) % n
        return self.wheel.pocket_order[idx]

    def zone(self, measured_index: float, width: int = 9) -> list[int]:
        """The `width` pockets centred on the named pocket, in wheel order."""
        if self.offset is None:
            raise RuntimeError("PocketMap is uncalibrated")
        n = self.wheel.n_pockets
        centre = int(round(measured_index + self.offset)) % n
        half = width // 2
        return [self.wheel.pocket_order[(centre + k) % n]
                for k in range(-half, width - half)]
