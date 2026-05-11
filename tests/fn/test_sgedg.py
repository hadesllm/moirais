"""Tests for edge correction."""
import numpy as np
from morie.fn.sgedg import sgedg


def test_sgedg_ripley():
    rng = np.random.default_rng(42)
    pts = rng.uniform(1, 9, (15, 2))
    r = sgedg(pts, (0, 10, 0, 10), method="ripley")
    assert r.name == "edge_correction"
    assert "weights" in r.extra
    assert r.extra["mean_border_dist"] > 0


def test_sgedg_border():
    rng = np.random.default_rng(42)
    pts = rng.uniform(1, 9, (15, 2))
    r = sgedg(pts, (0, 10, 0, 10), method="border")
    assert r.name == "edge_correction"
