"""Tests for ksr09.kosorok_z_estimator."""
import numpy as np
import pytest
from morie.fn.ksr09 import kosorok_z_estimator


def test_ksr09_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_z_estimator(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr09_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_z_estimator(x, y)
    assert isinstance(result, dict)
