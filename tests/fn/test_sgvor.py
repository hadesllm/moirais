"""Tests for Voronoi tessellation."""
import numpy as np
from moirais.fn.sgvor import sgvor


def test_sgvor_smoke():
    rng = np.random.default_rng(42)
    pts = rng.uniform(1, 9, (20, 2))
    r = sgvor(pts)
    assert r.name == "voronoi_tessellation"
    assert "cell_areas" in r.extra
    assert r.extra["n_finite_cells"] > 0


def test_cheatsheet():
    from moirais.fn.sgvor import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
