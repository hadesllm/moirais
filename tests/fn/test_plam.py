"""Tests for moirais.fn.plam — plot A-M summary."""
import numpy as np
from moirais.fn.plam import plam


def test_plam_smoke():
    rng = np.random.default_rng(42)
    pos = rng.standard_normal(50)
    r = plam(pos, bins=10)
    assert r.name == "plot_am_summary"
    assert len(r.extra["counts"]) == 10


def test_plam_with_weights():
    r = plam([1, 2, 3, 4], weights=[1, -1, 1, -1])
    assert r.extra["n_positive"] == 2
    assert r.extra["n_negative"] == 2
