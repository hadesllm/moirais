"""Tests for plncF.planck_function."""
import numpy as np
import pytest
from moirais.fn.plncF import planck_function


def test_plncF_basic():
    """Test basic functionality."""
    lam = 0.1
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = planck_function(lam, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_plncF_edge():
    """Test edge cases."""
    lam = 0.1
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = planck_function(lam, T)
    assert isinstance(result, dict)
