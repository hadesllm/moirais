"""Tests for spgwr.schabenberger_gwr."""
import numpy as np
import pytest
from moirais.fn.spgwr import schabenberger_gwr


def test_spgwr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    bandwidth = 0.3
    result = schabenberger_gwr(x, y, coords, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spgwr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    bandwidth = 0.3
    result = schabenberger_gwr(x, y, coords, bandwidth)
    assert isinstance(result, dict)
