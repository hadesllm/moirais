"""Tests for multi-scale quadrat aggregation."""
import numpy as np
from moirais.fn.sgqag import sgqag


def test_sgqag_smoke():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 10, (100, 2))
    r = sgqag(pts, [3, 5, 7], window=(0, 10, 0, 10))
    assert r.name == "quadrat_aggregation"
    assert "VMR_values" in r.extra
    assert len(r.extra["VMR_values"]) == 3


def test_cheatsheet():
    from moirais.fn.sgqag import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
