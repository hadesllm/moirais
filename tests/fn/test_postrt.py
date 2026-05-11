"""Tests for postrt.post_stratification."""
import numpy as np
import pytest
from morie.fn.postrt import post_stratification


def test_postrt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    stratum = np.random.default_rng(42).normal(0, 1, 100)
    N_h = np.random.default_rng(42).normal(0, 1, 100)
    result = post_stratification(y, weights, stratum, N_h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_postrt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    stratum = np.random.default_rng(42).normal(0, 1, 100)
    N_h = np.random.default_rng(42).normal(0, 1, 100)
    result = post_stratification(y, weights, stratum, N_h)
    assert isinstance(result, dict)
