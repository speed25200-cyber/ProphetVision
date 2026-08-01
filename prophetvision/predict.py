"""Landing zone prediction: fuses the fitted ball and rotor dynamics with the
learned scatter distribution into a probability over pockets."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import WheelConfig
from .geometry import wrap_deg
from .physics import BallDecayModel, RotorModel
from .scatter import ScatterModel
from .tracking import TrackResult


@dataclass
class Prediction:
    probabilities: np.ndarray        # P(final pocket index), length n
    wheel: WheelConfig
    impact_pocket_index: int         # deterministic rim-departure impact pocket
    t_drop: float                    # predicted time ball leaves the rim (s)
    t_impact: float                  # predicted first rotor contact time (s)
    ball_model: BallDecayModel = None
    rotor_model: RotorModel = None

    @property
    def top_pocket(self) -> int:
        return self.wheel.pocket_order[int(np.argmax(self.probabilities))]

    def best_zone(self, width: int = 9) -> tuple[list[int], float]:
        """Contiguous arc of `width` pockets with maximal total probability.

        Returns (pocket numbers in wheel order, total probability).
        """
        n = self.wheel.n_pockets
        p = self.probabilities
        best_start, best_mass = 0, -1.0
        for s in range(n):
            mass = float(sum(p[(s + j) % n] for j in range(width)))
            if mass > best_mass:
                best_start, best_mass = s, mass
        zone = [self.wheel.pocket_order[(best_start + j) % n]
                for j in range(width)]
        return zone, best_mass

    def summary(self, zone_width: int = 9) -> dict:
        zone, mass = self.best_zone(zone_width)
        order = np.argsort(self.probabilities)[::-1]
        return {
            "top_pocket": int(self.top_pocket),
            "top5": [(int(self.wheel.pocket_order[i]),
                      round(float(self.probabilities[i]), 4))
                     for i in order[:5]],
            "zone": zone,
            "zone_probability": round(mass, 4),
            "t_drop_s": round(self.t_drop, 3),
            "t_impact_s": round(self.t_impact, 3),
        }


@dataclass
class LandingZonePredictor:
    """Session-level predictor. Holds the wheel layout, the drop-phase
    parameters and the scatter model, and improves as spins are observed."""

    wheel: WheelConfig = field(default_factory=WheelConfig)
    # |angular speed| at which the ball can no longer hold the rim (deg/s).
    # ~1.6-2.2 rev/s on full-size wheels; refined online.
    omega_drop: float = 640.0
    # Time and azimuthal travel between leaving the rim and first rotor
    # contact (deflector region). Refined online.
    fall_time: float = 0.75
    fall_travel_deg: float = 120.0
    scatter: ScatterModel = None
    # Azimuth of the reference (zero) pocket at rotor-track time origin.
    rotor_zero_azimuth: float = 0.0

    _last_prediction: Prediction | None = None

    def __post_init__(self):
        if self.scatter is None:
            self.scatter = ScatterModel(n_pockets=self.wheel.n_pockets)

    # ------------------------------------------------------------------
    def predict(self, track: TrackResult) -> Prediction:
        ball = BallDecayModel.fit(track.t, track.theta_ball)
        rotor = RotorModel.fit(track.t_rotor, track.phi_rotor)

        t_drop = ball.time_to_speed(self.omega_drop)
        theta_drop = ball.theta_at(t_drop)
        t_impact = t_drop + self.fall_time
        theta_impact = theta_drop + ball.direction * self.fall_travel_deg

        # Reference pocket azimuth at impact time (image convention:
        # azimuth increases clockwise, matching clockwise pocket order).
        if track.phi_is_absolute:
            phi_zero = rotor.phi_at(t_impact)
        else:
            phi_zero = (self.rotor_zero_azimuth
                        + (rotor.phi_at(t_impact)
                           - rotor.phi_at(track.t_rotor[0])))
        rel = wrap_deg(theta_impact - phi_zero)
        n = self.wheel.n_pockets
        impact_idx = int(round(rel / (360.0 / n))) % n

        # Scatter offsets are expressed in the ball's direction of travel
        # relative to the rotor; map onto pocket-index direction.
        offsets = self.scatter.distribution()
        probs = np.zeros(n)
        sign = int(np.sign(ball.direction)) or 1
        for off in range(n):
            probs[(impact_idx + sign * off) % n] += offsets[off]
        probs /= probs.sum()

        pred = Prediction(
            probabilities=probs, wheel=self.wheel,
            impact_pocket_index=impact_idx,
            t_drop=t_drop, t_impact=t_impact,
            ball_model=ball, rotor_model=rotor,
        )
        self._last_prediction = pred
        return pred

    # ------------------------------------------------------------------
    def observe_outcome(self, final_pocket: int) -> None:
        """Feed the true outcome of the last predicted spin back into the
        scatter model (online learning)."""
        if self._last_prediction is None:
            return
        pred = self._last_prediction
        n = self.wheel.n_pockets
        final_idx = self.wheel.index_of(final_pocket)
        sign = int(np.sign(pred.ball_model.direction)) or 1
        offset = (sign * (final_idx - pred.impact_pocket_index)) % n
        self.scatter.observe(offset)
        self._last_prediction = None
