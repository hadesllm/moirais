"""Tests for fn/deff.py -- Design effect (DEFF)."""
import numpy as np

from moirais.fn.deff import deff, design_effect


def test_deff_equal_weights():
    """Equal weights should give DEFF = 1.0."""
    w = np.ones(100)
    result = deff(w)
    assert abs(result - 1.0) < 1e-10


def test_deff_unequal_weights():
    """Unequal weights should give DEFF > 1.0."""
    rng = np.random.default_rng(42)
    w = rng.uniform(1, 10, size=100)
    result = design_effect(w)
    assert result > 1.0


def test_deff_extreme_weights():
    """Very unequal weights should give large DEFF."""
    w = np.array([1.0, 1.0, 1.0, 100.0])
    result = deff(w)
    assert result > 2.0
