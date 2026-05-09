"""Tests for archtt.arch_test."""
import numpy as np
import pytest
from moirais.fn.archtt import arch_test


def test_archtt_basic():
    """Test basic functionality."""
    residuals = np.random.default_rng(42).normal(0, 1, 100)
    lags = 10
    result = arch_test(residuals, lags)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_archtt_edge():
    """Test edge cases."""
    residuals = np.random.default_rng(42).normal(0, 1, 100)
    lags = 10
    result = arch_test(residuals, lags)
    assert isinstance(result, dict)
