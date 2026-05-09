"""Tests for hmarim.geron_arima."""
import numpy as np
import pytest
from moirais.fn.hmarim import geron_arima


def test_hmarim_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    d = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_arima(y, p, d, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmarim_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    d = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_arima(y, p, d, q)
    assert isinstance(result, dict)
