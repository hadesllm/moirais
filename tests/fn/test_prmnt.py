"""Tests for permutation entropy."""
import numpy as np
import pytest
from moirais.fn.prmnt import permutation_entropy, prmnt


def test_sorted():
    x = np.arange(100, dtype=float)
    r = permutation_entropy(x, order=3)
    assert r.estimate == pytest.approx(0.0, abs=0.01)


def test_random():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 1000)
    r = permutation_entropy(x, order=3)
    assert r.estimate > 0.9


def test_alias():
    assert prmnt is permutation_entropy
