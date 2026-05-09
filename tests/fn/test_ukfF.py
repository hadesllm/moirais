"""Tests for ukfF.unscented_kalman."""
import numpy as np
import pytest
from moirais.fn.ukfF import unscented_kalman


def test_ukfF_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    Q = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = unscented_kalman(y, f, h, Q, R)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ukfF_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    Q = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = unscented_kalman(y, f, h, Q, R)
    assert isinstance(result, dict)
