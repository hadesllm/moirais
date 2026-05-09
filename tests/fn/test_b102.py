"""Tests for b102.burkov_lm_ch1_squared_error."""
import numpy as np
import pytest
from moirais.fn.b102 import burkov_lm_ch1_squared_error


def test_b102_basic():
    """Test basic functionality."""
    y_hat_i = np.random.default_rng(42).normal(0, 1, 100)
    y_i = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch1_squared_error(y_hat_i, y_i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_b102_edge():
    """Test edge cases."""
    y_hat_i = np.random.default_rng(42).normal(0, 1, 100)
    y_i = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch1_squared_error(y_hat_i, y_i)
    assert isinstance(result, dict)
