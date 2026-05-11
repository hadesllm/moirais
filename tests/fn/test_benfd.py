"""Tests for benfd (Benford law test)."""
import numpy as np
from morie.fn.benfd import benfords_law_test


def test_benfords_basic():
    rng = np.random.default_rng(42)
    data = rng.exponential(100, size=1000)
    r = benfords_law_test(data)
    assert "chi2_statistic" in r.extra
    assert "p_value" in r.extra


def test_cheatsheet():
    from morie.fn.benfd import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
