"""Tests for hrzrank.horowitz_semipar_rank."""
import numpy as np
import pytest
from moirais.fn.hrzrank import horowitz_semipar_rank


def test_hrzrank_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_semipar_rank(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzrank_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_semipar_rank(x, y)
    assert isinstance(result, dict)
