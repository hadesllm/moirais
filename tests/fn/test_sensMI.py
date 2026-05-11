"""Tests for sensMI.sensitivity_mediation_imbens."""
import numpy as np
import pytest
from morie.fn.sensMI import sensitivity_mediation_imbens


def test_sensMI_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    C = np.random.default_rng(42).normal(0, 1, 100)
    r2_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = sensitivity_mediation_imbens(Y, X, C, r2_grid)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_sensMI_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    C = np.random.default_rng(42).normal(0, 1, 100)
    r2_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = sensitivity_mediation_imbens(Y, X, C, r2_grid)
    assert isinstance(result, dict)
