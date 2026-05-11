"""Tests for morie.fn.outam — A-M outlier detection."""
import numpy as np
from morie.fn.outam import outam


def test_outam_smoke():
    rng = np.random.default_rng(42)
    R = rng.standard_normal((20, 5)) * 0.1
    R[0] = 10.0  # inject outlier
    r = outam(R, threshold=2.0)
    assert r.name == "outlier_detection_am"
    assert 0 in r.extra["outlier_indices"]


def test_cheatsheet():
    from morie.fn.outam import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
