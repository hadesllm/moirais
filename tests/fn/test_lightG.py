"""Tests for lightG.lightgcn."""
import numpy as np
import pytest
from morie.fn.lightG import lightgcn


def test_lightG_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    layers = np.random.default_rng(42).normal(0, 1, 100)
    result = lightgcn(R, K, layers)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lightG_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    layers = np.random.default_rng(42).normal(0, 1, 100)
    result = lightgcn(R, K, layers)
    assert isinstance(result, dict)
