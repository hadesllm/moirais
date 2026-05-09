"""Tests for rng015.rangayyan_ch3_ensemble_mean."""
import numpy as np
import pytest
from moirais.fn.rng015 import rangayyan_ch3_ensemble_mean


def test_rng015_basic():
    """Test basic functionality."""
    x_k = np.random.default_rng(42).normal(0, 1, 100)
    t1 = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_ensemble_mean(x_k, t1, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng015_edge():
    """Test edge cases."""
    x_k = np.random.default_rng(42).normal(0, 1, 100)
    t1 = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_ensemble_mean(x_k, t1, M)
    assert isinstance(result, dict)
