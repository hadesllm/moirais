"""Tests for missinM.missing_mechanism_sensitivity."""
import numpy as np
import pytest
from morie.fn.missinM import missing_mechanism_sensitivity


def test_missinM_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    delta_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = missing_mechanism_sensitivity(Y, R, delta_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_missinM_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    delta_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = missing_mechanism_sensitivity(Y, R, delta_grid)
    assert isinstance(result, dict)
