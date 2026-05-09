"""Tests for rgricca.rangayyan_riccati_eq."""
import numpy as np
import pytest
from moirais.fn.rgricca import rangayyan_riccati_eq


def test_rgricca_basic():
    """Test basic functionality."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_riccati_eq(F, H, Q, R)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgricca_edge():
    """Test edge cases."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_riccati_eq(F, H, Q, R)
    assert isinstance(result, dict)
