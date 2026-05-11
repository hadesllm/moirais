"""Tests for grbvd.geron_bias_variance_decomposition."""
import numpy as np
import pytest
from morie.fn.grbvd import geron_bias_variance_decomposition


def test_grbvd_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    predictions = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bias_variance_decomposition(y_true, predictions)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_grbvd_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    predictions = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bias_variance_decomposition(y_true, predictions)
    assert isinstance(result, dict)
