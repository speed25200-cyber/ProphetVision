"""Shared fixtures for the live/dashboard tests (SPEC F-H).

The LiveEngine session on clip A takes ~45 s; a session-scoped fixture
runs it once for both test modules.  All real-video tests skip gracefully
when the video windows are unavailable.  Video directory can be overridden
with the PV_VIDEOS environment variable.
"""
from __future__ import annotations

import os

import pytest

VIDEOS = os.environ.get("PV_VIDEOS", "/mnt/agents/output/prophetvision/videos")
CLIP_A = os.path.join(VIDEOS, "hq_060_100.mp4")
CLIP_B = os.path.join(VIDEOS, "hq_100_135.mp4")
CLIP_720 = os.path.join(VIDEOS, "video720.mp4")

has_clip_a = os.path.exists(CLIP_A)
has_clip_720 = os.path.exists(CLIP_720)


@pytest.fixture(scope="session")
def session_a():
    """(SessionReport, peak-RAM delta MB) of the engine on clip A."""
    import resource
    from prophetvision.live import LiveEngine
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rep = LiveEngine().run(CLIP_A)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rep, (after - before) / 1024.0  # ru_maxrss is KB on Linux
