"""Full-pipeline validation on physically simulated synthetic spins.

The predictor sees only the early part of each spin (cut off well before the
ball leaves the rim) and must predict the landing zone; the simulation's
ground truth says where the ball actually settled.
"""

import numpy as np
import pytest

from prophetvision.pipeline import analyze_frames
from prophetvision.predict import LandingZonePredictor
from prophetvision.synthetic import SyntheticSpin


def run_spin(seed: int, predictor=None, lead: float = 2.0):
    spin = SyntheticSpin(seed=seed)
    frames, times, truth = spin.simulate()
    cutoff = truth.t_drop - lead
    assert cutoff > 1.0, "spin too short for the requested lead time"
    pred, track, predictor = analyze_frames(
        frames, spin.fps, predictor=predictor, cutoff_seconds=cutoff)
    return spin, truth, pred, predictor


def circular_err(a, b, n=37):
    d = (a - b) % n
    return min(d, n - d)


def test_impact_pocket_prediction_accuracy():
    """Rim-departure impact point predicted 2 s ahead, within a few pockets."""
    errs = []
    for seed in range(4):
        spin, truth, pred, _ = run_spin(seed)
        errs.append(circular_err(pred.impact_pocket_index, truth.impact_index))
    assert np.median(errs) <= 4, f"impact errors too large: {errs}"


def test_drop_time_prediction():
    spin, truth, pred, _ = run_spin(0)
    assert abs(pred.t_drop - truth.t_drop) < 0.5


def test_zone_hit_rate_beats_chance():
    """A 9-pocket zone covers 24% of the wheel; the predictor must hit it
    far more often than chance across seeds."""
    hits = 0
    n_spins = 6
    predictor = None
    for seed in range(n_spins):
        spin, truth, pred, predictor = run_spin(seed, predictor=predictor)
        zone, _ = pred.best_zone(9)
        hits += truth.final_pocket in zone
        predictor.observe_outcome(truth.final_pocket)  # online learning
    assert hits >= 3, f"only {hits}/{n_spins} zone hits (chance ~1.5)"


def test_online_learning_updates_scatter():
    predictor = None
    spin, truth, pred, predictor = run_spin(0)
    before = predictor.scatter.n_observations()
    predictor.observe_outcome(truth.final_pocket)
    assert predictor.scatter.n_observations() == pytest.approx(before + 1)
