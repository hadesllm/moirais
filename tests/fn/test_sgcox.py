"""Tests for Cox process."""
import numpy as np
from moirais.fn.sgcox import sgcox


def test_sgcox_smoke():
    rng = np.random.default_rng(42)
    intensity = rng.uniform(1, 10, (10, 10))
    r = sgcox(intensity, (0, 10, 0, 10), seed=42)
    assert r.name == "cox_process"
    assert "points" in r.extra
    assert r.extra["n_points"] > 0


def test_cheatsheet():
    from moirais.fn.sgcox import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
