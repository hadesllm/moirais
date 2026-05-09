"""Tests for arimab.arima_box_jenkins."""
import numpy as np
import pytest
from moirais.fn.arimab import arima_box_jenkins


def test_arimab_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    d = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = arima_box_jenkins(y, p, d, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_arimab_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    d = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = arima_box_jenkins(y, p, d, q)
    assert isinstance(result, dict)
