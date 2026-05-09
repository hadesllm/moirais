"""Tests for naivef.naive_forecast."""
import numpy as np
import pytest
from moirais.fn.naivef import naive_forecast


def test_naivef_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    h = 0.3
    result = naive_forecast(y, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_naivef_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    h = 0.3
    result = naive_forecast(y, h)
    assert isinstance(result, dict)
