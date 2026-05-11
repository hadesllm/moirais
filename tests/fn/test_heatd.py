"""Tests for morie.fn.heatd."""
import numpy as np
from morie.fn.heatd import heatd


def test_heatd_smoke():
    rng = np.random.default_rng(42)
    result = heatd(lat=rng.uniform(40, 45, size=20), lon=rng.uniform(-80, -75, size=20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.heatd import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
