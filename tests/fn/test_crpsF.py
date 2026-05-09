"""Tests for crpsF.crps."""
import numpy as np
import pytest
from moirais.fn.crpsF import crps


def test_crpsF_basic():
    """Test basic functionality."""
    forecast_cdf = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = crps(forecast_cdf, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crpsF_edge():
    """Test edge cases."""
    forecast_cdf = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = crps(forecast_cdf, y)
    assert isinstance(result, dict)
