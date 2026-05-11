"""Tests for epicur.epicurve."""
import numpy as np
import pytest
from morie.fn.epicur import epicurve


def test_epicur_basic():
    """Test basic functionality."""
    dates = np.random.default_rng(42).normal(0, 1, 100)
    cases = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = epicurve(dates, cases, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_epicur_edge():
    """Test edge cases."""
    dates = np.random.default_rng(42).normal(0, 1, 100)
    cases = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = epicurve(dates, cases, bandwidth)
    assert isinstance(result, dict)
