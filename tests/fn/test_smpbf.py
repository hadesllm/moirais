"""Tests for morie.fn.smpbf."""
import numpy as np
from morie.fn.smpbf import smpbf


def test_smpbf_smoke():
    rng = np.random.default_rng(42)
    result = smpbf(
        lat=rng.uniform(40, 45, size=20),
        lon=rng.uniform(-80, -75, size=20),
        radius_km=1.0
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.smpbf import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
