"""Tests for spwls.schabenberger_wls_variogram."""
import numpy as np
import pytest
from moirais.fn.spwls import schabenberger_wls_variogram


def test_spwls_basic():
    """Test basic functionality."""
    empirical_variogram = np.random.default_rng(42).normal(0, 1, 100)
    variogram_model = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_wls_variogram(empirical_variogram, variogram_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spwls_edge():
    """Test edge cases."""
    empirical_variogram = np.random.default_rng(42).normal(0, 1, 100)
    variogram_model = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_wls_variogram(empirical_variogram, variogram_model)
    assert isinstance(result, dict)
