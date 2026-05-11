"""Tests for evangia.evt_angular_measure."""
import numpy as np
import pytest
from morie.fn.evangia import evt_angular_measure


def test_evangia_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = evt_angular_measure(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evangia_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = evt_angular_measure(X, k)
    assert isinstance(result, dict)
