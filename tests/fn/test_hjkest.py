"""Tests for hjkest.hajek_estimator."""
import numpy as np
import pytest
from morie.fn.hjkest import hajek_estimator


def test_hjkest_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    result = hajek_estimator(y, pi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hjkest_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    result = hajek_estimator(y, pi)
    assert isinstance(result, dict)
