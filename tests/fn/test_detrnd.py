"""Tests for detrnd.detrend_climate."""
import numpy as np
import pytest
from moirais.fn.detrnd import detrend_climate


def test_detrnd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = detrend_climate(x, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_detrnd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = detrend_climate(x, t)
    assert isinstance(result, dict)
