"""Tests for homogeneous Poisson process."""
import numpy as np
from morie.fn.sghpp import sghpp


def test_sghpp_smoke():
    r = sghpp(50, (0, 10, 0, 10), seed=42)
    assert r.name == "homogeneous_poisson"
    assert "points" in r.extra
    assert r.extra["points"].shape[1] == 2


def test_sghpp_in_window():
    r = sghpp(30, (0, 5, 0, 5), seed=42)
    pts = r.extra["points"]
    assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] <= 5)
