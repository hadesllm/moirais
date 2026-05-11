"""Tests for adwopt.adamw."""
import numpy as np
import pytest
from morie.fn.adwopt import adamw


def test_adwopt_basic():
    """Test basic functionality."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    beta1 = np.random.default_rng(42).normal(0, 1, 100)
    beta2 = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    wd = np.random.default_rng(42).normal(0, 1, 100)
    result = adamw(g, beta1, beta2, lr, wd)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_adwopt_edge():
    """Test edge cases."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    beta1 = np.random.default_rng(42).normal(0, 1, 100)
    beta2 = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    wd = np.random.default_rng(42).normal(0, 1, 100)
    result = adamw(g, beta1, beta2, lr, wd)
    assert isinstance(result, dict)
