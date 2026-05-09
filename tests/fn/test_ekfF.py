"""Tests for ekfF.extended_kalman."""
import numpy as np
import pytest
from moirais.fn.ekfF import extended_kalman


def test_ekfF_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    F = np.random.default_rng(43).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = extended_kalman(y, f, h, F, H, Q, R)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ekfF_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    F = np.random.default_rng(43).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = extended_kalman(y, f, h, F, H, Q, R)
    assert isinstance(result, dict)
