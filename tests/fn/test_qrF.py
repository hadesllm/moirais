"""Tests for qrF.quantile_forecast."""
import numpy as np
import pytest
from morie.fn.qrF import quantile_forecast


def test_qrF_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    y_hat = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = quantile_forecast(y, y_hat, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_qrF_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    y_hat = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = quantile_forecast(y, y_hat, tau)
    assert isinstance(result, dict)
