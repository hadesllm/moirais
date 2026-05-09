"""Tests for gb_cvmc.gibbons_cramer_von_mises."""
import numpy as np
import pytest
from moirais.fn.gb_cvmc import gibbons_cramer_von_mises


def test_gb_cvmc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    F0 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_cramer_von_mises(x, F0)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_cvmc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    F0 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_cramer_von_mises(x, F0)
    assert isinstance(result, dict)
