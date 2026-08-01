"""ProphetVision — physics-informed ball landing zone prediction from video.

Pipeline: video -> calibration -> ball/rotor tracking -> physics model fit
-> drop-point prediction -> scatter model -> landing zone probabilities.
"""

__version__ = "0.1.0"

from .config import WheelConfig, EUROPEAN_ORDER, AMERICAN_ORDER
from .physics import BallDecayModel, RotorModel
from .scatter import ScatterModel
from .predict import LandingZonePredictor, Prediction
from .live import LiveEngine, SessionReport

__all__ = [
    "WheelConfig",
    "EUROPEAN_ORDER",
    "AMERICAN_ORDER",
    "BallDecayModel",
    "RotorModel",
    "ScatterModel",
    "LandingZonePredictor",
    "Prediction",
    "LiveEngine",
    "SessionReport",
]
