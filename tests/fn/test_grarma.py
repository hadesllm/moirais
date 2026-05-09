"""Tests for grarma.geron_arima_forecast."""
import numpy as np
import pytest
from moirais.fn.grarma import geron_arima_forecast


def test_grarma_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    phi = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    d = 5
    result = geron_arima_forecast(y, phi, theta, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grarma_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    phi = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    d = 5
    result = geron_arima_forecast(y, phi, theta, d)
    assert isinstance(result, dict)
