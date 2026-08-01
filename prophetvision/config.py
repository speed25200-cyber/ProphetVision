"""Wheel geometry and layout configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

# Pocket number order, clockwise when viewed from above.
EUROPEAN_ORDER = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23,
    10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26,
]

AMERICAN_ORDER = [
    0, 28, 9, 26, 30, 11, 7, 20, 32, 17, 5, 22, 34, 15, 3, 24, 36, 13,
    1, 37, 27, 10, 25, 29, 12, 8, 19, 31, 18, 6, 21, 33, 16, 4, 23, 35,
    14, 2,
]  # 37 represents "00"


@dataclass
class WheelConfig:
    """Geometry and layout of the wheel as it appears in the video.

    Radii are fractions of the outer wheel radius so a single calibration
    (center + outer radius in pixels) fully determines the geometry.
    """

    pocket_order: list[int] = field(default_factory=lambda: list(EUROPEAN_ORDER))
    # Annulus where the ball circulates while on the rim (fractions of R).
    ball_track_inner: float = 0.82
    ball_track_outer: float = 1.02
    # Radius of the ring used to track rotor rotation (number/pocket ring).
    rotor_ring_radius: float = 0.55
    rotor_ring_width: float = 0.10
    # True if the wheel spins clockwise in the image, ball counter-clockwise
    # (the usual arrangement). Tracking estimates signs itself; these are
    # only used by the synthetic generator.
    clockwise_rotor: bool = True

    @property
    def n_pockets(self) -> int:
        return len(self.pocket_order)

    def pocket_at_angle(self, rel_angle_deg: float) -> int:
        """Pocket number located at `rel_angle_deg` degrees from the rotor's
        reference mark (pocket_order[0], e.g. the green zero), measured in
        the direction of increasing pocket index."""
        n = self.n_pockets
        idx = int(round(rel_angle_deg / (360.0 / n))) % n
        return self.pocket_order[idx]

    def index_of(self, pocket: int) -> int:
        return self.pocket_order.index(pocket)
