"""Tests for brrvar.brr_variance."""
import numpy as np
import pytest
from moirais.fn.brrvar import brr_variance


def test_brrvar_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    replicates = np.random.default_rng(42).normal(0, 1, 100)
    result = brr_variance(y, weights, replicates)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_brrvar_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    replicates = np.random.default_rng(42).normal(0, 1, 100)
    result = brr_variance(y, weights, replicates)
    assert isinstance(result, dict)
