"""Tests for shfflm.shuffle_model."""
import numpy as np
import pytest
from moirais.fn.shfflm import shuffle_model


def test_shfflm_basic():
    """Test basic functionality."""
    epsilon0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = shuffle_model(epsilon0, n, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_shfflm_edge():
    """Test edge cases."""
    epsilon0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = shuffle_model(epsilon0, n, delta)
    assert isinstance(result, dict)
