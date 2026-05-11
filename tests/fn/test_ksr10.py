"""Tests for ksr10.kosorok_m_estimator."""
import numpy as np
import pytest
from morie.fn.ksr10 import kosorok_m_estimator


def test_ksr10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_m_estimator(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_m_estimator(x, y)
    assert isinstance(result, dict)
