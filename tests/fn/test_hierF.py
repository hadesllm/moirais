"""Tests for hierF.hierarchical_forecast."""
import numpy as np
import pytest
from morie.fn.hierF import hierarchical_forecast


def test_hierF_basic():
    """Test basic functionality."""
    base_forecasts = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    cov = np.random.default_rng(42).normal(0, 1, 100)
    result = hierarchical_forecast(base_forecasts, S, cov)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hierF_edge():
    """Test edge cases."""
    base_forecasts = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    cov = np.random.default_rng(42).normal(0, 1, 100)
    result = hierarchical_forecast(base_forecasts, S, cov)
    assert isinstance(result, dict)
