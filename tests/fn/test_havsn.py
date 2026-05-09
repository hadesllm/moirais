"""Tests for moirais.fn.havsn."""
import numpy as np
from moirais.fn.havsn import havsn


def test_havsn_smoke():
    rng = np.random.default_rng(42)
    result = havsn(lat1=43.65, lon1=-79.38, lat2=43.7, lon2=-79.4)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.havsn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
