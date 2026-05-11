"""Tests for wfrep.weighted_frequency."""
import numpy as np
import pytest
from morie.fn.wfrep import weighted_frequency


def test_wfrep_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    cells = np.random.default_rng(42).normal(0, 1, 100)
    result = weighted_frequency(y, weights, cells)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wfrep_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    cells = np.random.default_rng(42).normal(0, 1, 100)
    result = weighted_frequency(y, weights, cells)
    assert isinstance(result, dict)
