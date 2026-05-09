"""Tests for spols.schabenberger_ols_variogram."""
import numpy as np
import pytest
from moirais.fn.spols import schabenberger_ols_variogram


def test_spols_basic():
    """Test basic functionality."""
    empirical_variogram = np.random.default_rng(42).normal(0, 1, 100)
    variogram_model = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_ols_variogram(empirical_variogram, variogram_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spols_edge():
    """Test edge cases."""
    empirical_variogram = np.random.default_rng(42).normal(0, 1, 100)
    variogram_model = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_ols_variogram(empirical_variogram, variogram_model)
    assert isinstance(result, dict)
