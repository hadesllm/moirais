"""Tests for lag class binning."""
import numpy as np
from morie.fn.sglgc import sglgc


def test_sglgc_smoke():
    dists = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    r = sglgc(dists, n_bins=5)
    assert r.name == "lag_class_binning"
    assert "bin_counts" in r.extra
    assert len(r.extra["bin_counts"]) == 5


def test_cheatsheet():
    from morie.fn.sglgc import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
