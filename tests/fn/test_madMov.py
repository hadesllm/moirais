"""Tests for madMov.moving_mad."""
import numpy as np
import pytest
from moirais.fn.madMov import moving_mad


def test_madMov_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = moving_mad(x, window, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_madMov_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = moving_mad(x, window, k)
    assert isinstance(result, dict)
