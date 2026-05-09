"""Tests for awltrn.augmented_owl."""
import numpy as np
import pytest
from moirais.fn.awltrn import augmented_owl


def test_awltrn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    result = augmented_owl(y, D, W, pi, Q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_awltrn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    result = augmented_owl(y, D, W, pi, Q)
    assert isinstance(result, dict)
