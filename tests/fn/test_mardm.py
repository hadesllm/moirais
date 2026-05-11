"""Test mardm."""
import numpy as np
import pytest
from morie.fn.mardm import mardia_test


def test_mardm_basic():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 3))
    r = mardia_test(X)
    assert r.test_name == "Mardia"
    assert r.p_value >= 0


def test_mardm_nonnormal():
    rng = np.random.default_rng(7)
    X = rng.exponential(1.0, (80, 2))
    r = mardia_test(X)
    assert r.statistic >= 0
    assert "kurtosis_b2p" in r.extra
