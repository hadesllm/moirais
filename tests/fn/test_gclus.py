"""Tests for moirais.fn.gclus."""
import numpy as np
from moirais.fn.gclus import gclus


def test_gclus_smoke():
    rng = np.random.default_rng(42)
    result = gclus(lat=rng.uniform(40, 45, size=20), lon=rng.uniform(-80, -75, size=20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.gclus import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
