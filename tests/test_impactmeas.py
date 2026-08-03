import numpy as np
import pytest

from prophetvision.impactmeas import (ImpactResult, bounce_spread,
                                      measure_impact)

ROTOR = 9.0          # pockets/s, rotor turning one way
BALL_LAB = -6.0      # the ball runs the other way on the rim


POCKET = 360.0 / 37


def _arc(t0, t1, lab_rate, index0, dt=0.02, r=0.85, conf=0.8, rotor=ROTOR,
         r_end=None):
    """Rim detections for an object moving at ``lab_rate`` in the lab frame.

    Azimuth is what the camera sees and index is what the rotor sees, so they
    move at different rates — that difference is the whole discriminator and a
    fixture that collapses them would test nothing.
    """
    t = np.arange(t0, t1, dt)
    idx = (index0 + (lab_rate - rotor) * (t - t0)) % 37.0
    az = (index0 * POCKET + lab_rate * POCKET * (t - t0)) % 360.0
    rad = np.full(len(t), r) if r_end is None else np.linspace(r, r_end, len(t))
    return np.column_stack([t, az, rad, np.full(len(t), conf), idx])


def test_a_fixed_artefact_is_not_mistaken_for_the_ball():
    """The failure that corrupted the first bounce measurements: a reflection
    sitting still in the image drifts in the rotor frame at exactly the rotor's
    rate, and every relative-motion test accepts it."""
    static = _arc(10.0, 11.0, lab_rate=0.0, index0=20.0)
    assert measure_impact(static, ROTOR) is None


def test_the_ball_arc_is_accepted_and_read_at_its_end():
    ball = _arc(10.0, 11.0, lab_rate=BALL_LAB, index0=30.0, r=1.0, r_end=0.80)
    res = measure_impact(ball, ROTOR)
    assert res is not None
    assert res.lab_rate_pockets_s == pytest.approx(BALL_LAB, abs=0.2)
    assert res.trustworthy
    expected = (30.0 + (BALL_LAB - ROTOR) * (ball[-1, 0] - ball[0, 0])) % 37.0
    assert res.index == pytest.approx(expected, abs=0.3)


def test_the_ball_is_found_even_when_a_static_artefact_outlasts_it():
    """The artefact is present after the ball has gone, so 'take the latest
    arc' alone would return the artefact."""
    ball = _arc(10.0, 11.0, lab_rate=BALL_LAB, index0=30.0)
    static = _arc(11.3, 12.5, lab_rate=0.0, index0=5.0)
    res = measure_impact(np.vstack([ball, static]), ROTOR)
    assert res is not None
    assert res.t == pytest.approx(ball[-1, 0], abs=1e-6)
    assert res.n_runs == 1


def test_ball_and_artefact_present_at_the_same_instants_are_separated():
    """Observed on spin 1: a fixed highlight at azimuth 224 sits on the rim,
    frame for frame, right through the ball's arc. Treating the detections as
    one time-ordered series interleaves the two objects and the fitted rate
    means nothing."""
    from prophetvision.impactmeas import link_tracks
    ball = _arc(10.0, 11.0, lab_rate=BALL_LAB, index0=30.0)
    static = _arc(10.0, 11.0, lab_rate=0.0, index0=20.0)
    mixed = np.vstack([ball, static])
    tracks = link_tracks(mixed[mixed[:, 2] >= 0.78])
    assert len(tracks) == 2
    assert {len(a) for a in tracks} == {len(ball)}
    res = measure_impact(mixed, ROTOR)
    assert res is not None
    assert res.lab_rate_pockets_s == pytest.approx(BALL_LAB, abs=0.3)
    assert res.index == pytest.approx(ball[-1, 4], abs=0.3)


def test_a_ball_running_with_the_rotor_is_rejected():
    """On the rim the ball always runs against the rotor; anything co-rotating
    is some other object."""
    wrong_way = _arc(10.0, 11.0, lab_rate=+6.0, index0=12.0)
    assert measure_impact(wrong_way, ROTOR) is None


def test_index_rate_is_fitted_through_the_wrap():
    """The arc crosses index 0; without unwrapping the fitted rate collapses."""
    ball = _arc(10.0, 11.2, lab_rate=BALL_LAB, index0=3.0)
    assert (np.diff(ball[:, 4]) > 0).any()      # it really does wrap
    res = measure_impact(ball, ROTOR)
    assert res is not None
    assert res.rel_rate_pockets_s == pytest.approx(BALL_LAB - ROTOR, abs=0.3)


def test_only_the_last_contiguous_run_of_the_ball_is_used():
    """A gap means re-acquisition; bridging it can splice two objects."""
    early = _arc(10.0, 10.5, lab_rate=BALL_LAB, index0=30.0)
    late = _arc(11.5, 12.2, lab_rate=BALL_LAB, index0=8.0)
    res = measure_impact(np.vstack([early, late]), ROTOR)
    assert res is not None
    assert res.t == pytest.approx(late[-1, 0], abs=1e-6)
    assert res.arc_span_s < 0.8


def test_the_impact_is_read_where_the_ball_stops_running():
    """Spin 1 in the reference footage: the ball holds -92 deg/s, strikes the
    top deflector and stalls, and the detector follows the falling blob for
    another 0.3 s. The rotor turns ~3.7 pockets in that time, so taking the
    last sample of the track would bias the impact by that much."""
    ball = _arc(10.0, 11.0, lab_rate=BALL_LAB, index0=30.0, r=1.0, r_end=0.85)
    stall_t = np.arange(11.0, 11.35, 0.02)
    idx = (ball[-1, 4] - ROTOR * (stall_t - ball[-1, 0])) % 37.0
    stall = np.column_stack([stall_t, np.full(len(stall_t), ball[-1, 1]),
                             np.linspace(0.85, 0.79, len(stall_t)),
                             np.full(len(stall_t), 0.8), idx])
    res = measure_impact(np.vstack([ball, stall]), ROTOR)
    assert res is not None
    assert res.t == pytest.approx(ball[-1, 0], abs=0.06)
    assert res.exit_lag_s > 0.25
    assert res.index == pytest.approx(ball[-1, 4], abs=0.6)
    assert res.trustworthy


def test_a_late_fragment_does_not_outrank_the_real_arc():
    """Spin A: a nine-sample fragment half a second after the ball had landed
    beat the eighty-seven-sample arc under a 'take the latest' rule."""
    ball = _arc(10.0, 11.6, lab_rate=BALL_LAB, index0=30.0, r=0.80)
    frag = _arc(12.1, 12.28, lab_rate=BALL_LAB, index0=17.0, r=0.79)
    res = measure_impact(np.vstack([ball, frag]), ROTOR)
    assert res is not None
    assert res.t == pytest.approx(ball[-1, 0], abs=0.06)
    assert res.arc_samples > 50


def test_an_arc_still_high_on_the_rim_is_not_the_impact():
    """The ball spirals down; a track that ends at full radius is the ball in
    mid-flight, not where it left the rim."""
    high = _arc(10.0, 11.0, lab_rate=BALL_LAB, index0=30.0, r=1.00)
    assert measure_impact(high, ROTOR) is None
    low = _arc(11.4, 12.2, lab_rate=-4.5, index0=12.0, r=0.84)
    res = measure_impact(np.vstack([high, low]), ROTOR)
    assert res is not None
    assert res.t == pytest.approx(low[-1, 0], abs=0.06)


def test_detections_off_the_rim_are_ignored():
    inner = _arc(10.0, 11.0, lab_rate=BALL_LAB, index0=30.0, r=0.55)
    assert measure_impact(inner, ROTOR) is None


def test_starved_input_returns_none_rather_than_a_guess():
    assert measure_impact(np.zeros((3, 5)), ROTOR) is None
    with pytest.raises(ValueError):
        measure_impact(np.zeros((10, 3)), ROTOR)


def test_a_thin_arc_is_returned_but_not_trustworthy():
    thin = _arc(10.0, 10.16, lab_rate=BALL_LAB, index0=30.0, dt=0.02)
    res = measure_impact(thin, ROTOR)
    assert res is not None
    assert not res.trustworthy          # 8 samples, 0.14 s


def test_bounce_spread_is_blind_to_the_numbering_offset():
    """The whole point of the anchor-free estimate: the unknown vision-to-wheel
    offset shifts every v_i equally, so the spread does not depend on it."""
    from prophetvision.config import WheelConfig
    w = WheelConfig()
    rng = np.random.default_rng(3)
    results = [27, 25, 9, 1, 31, 14, 8]
    for offset in (0.0, 7.0, 19.0):
        impacts = [(w.index_of(p) - 4.0 - offset + rng.normal(0, 2.0)) % 37
                   for p in results]
        st = bounce_spread(impacts, results)
        assert st["sigma_pockets"] < 4.0
        assert st["n"] == len(results)


def test_bounce_spread_flags_a_uniform_scatter():
    rng = np.random.default_rng(0)
    results = [27, 25, 9, 1, 31, 14, 8, 23, 5, 36, 12, 30]
    impacts = list(rng.uniform(0, 37, len(results)))
    st = bounce_spread(impacts, results)
    assert st["rayleigh_p"] > 0.05           # no significant concentration
    assert st["sigma_pockets"] > 6.0


def test_trustworthy_needs_all_of_its_conditions():
    ok = dict(index=5.0, t=1.0, azimuth=271.0, confidence=0.8, radius=0.85,
              arc_samples=30,
              arc_span_s=0.6, rel_rate_pockets_s=-15.0,
              lab_rate_pockets_s=-6.0, rotor_pockets_s=9.0, n_runs=1,
              descent=0.20)
    assert ImpactResult(**ok).trustworthy
    assert not ImpactResult(**{**ok, "arc_samples": 8}).trustworthy
    assert not ImpactResult(**{**ok, "descent": 0.01}).trustworthy
    assert not ImpactResult(**{**ok, "lab_rate_pockets_s": -1.0}).trustworthy
    assert not ImpactResult(**{**ok, "lab_rate_pockets_s": 6.0}).trustworthy
    assert not ImpactResult(**{**ok, "confidence": 0.3}).trustworthy
