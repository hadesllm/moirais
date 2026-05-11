"""Tests for morie.fn.lrank — Log-rank test."""
import numpy as np

from morie.fn.lrank import log_rank, lrank


def test_same_distribution():
    """Same distribution: should not reject."""
    rng = np.random.default_rng(42)
    t1 = rng.exponential(5, size=100)
    e1 = np.ones(100, dtype=int)
    t2 = rng.exponential(5, size=100)
    e2 = np.ones(100, dtype=int)
    result = log_rank(t1, e1, t2, e2)
    assert result.p_value > 0.05


def test_different_distributions():
    """Very different distributions: should reject."""
    rng = np.random.default_rng(42)
    t1 = rng.exponential(2, size=100)
    e1 = np.ones(100, dtype=int)
    t2 = rng.exponential(10, size=100)
    e2 = np.ones(100, dtype=int)
    result = log_rank(t1, e1, t2, e2)
    assert result.p_value < 0.05


def test_lrank_alias():
    assert lrank is log_rank
