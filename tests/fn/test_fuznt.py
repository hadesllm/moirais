"""Tests for fuzzy entropy."""
import numpy as np
from morie.fn.fuznt import fuzzy_entropy, fuznt


def test_basic():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    r = fuzzy_entropy(x, m=2)
    assert r.estimate >= 0 or r.estimate == float("inf")


def test_alias():
    assert fuznt is fuzzy_entropy
