import numpy as np
import pytest

from prophetvision.balldetect import BallDetector, Detection, radial_gate


def test_radial_gate_keeps_annulus_only():
    accept = radial_gate(100.0, 50.0, 80.0, 30.0, 0.8, 1.05)
    assert accept(100.0 + 80.0, 50.0)          # on the ellipse, r = 1
    assert not accept(100.0, 50.0)             # hub, r = 0 -> rejected
    assert not accept(100.0 + 120.0, 50.0)     # far outside


def test_evaluate_reports_hit_rate_and_error():
    ref = [(1.0, 100.0, 100.0), (1.1, 110.0, 100.0), (1.2, 120.0, 100.0)]
    dets = [Detection(t=1.0, x=102.0, y=100.0, confidence=0.8),
            Detection(t=1.1, x=300.0, y=300.0, confidence=0.9)]
    out = BallDetector.evaluate(dets, ref, tol_px=25.0)
    assert out["instants"] == 3
    assert out["hits"] == 1                     # only the t=1.0 detection is close
    assert out["median_error_px"] > 2.0


def test_evaluate_handles_empty_detections():
    out = BallDetector.evaluate([], [(0.0, 1.0, 2.0)])
    assert out["hits"] == 0 and out["hit_rate"] == 0.0


def test_stride_must_leave_no_blind_band():
    """A stride wider than the interior of a tile leaves pixels that every
    tile sees only inside its rejected border."""
    pytest.importorskip("onnxruntime")
    with pytest.raises(ValueError):
        BallDetector(model_path="unused.onnx", tile=320, stride=300, border=26)
