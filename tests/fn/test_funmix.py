"""Tests for funmix.functional_mixture."""
import numpy as np
import pytest
from morie.fn.funmix import functional_mixture


def test_funmix_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = functional_mixture(Y, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_funmix_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = functional_mixture(Y, K)
    assert isinstance(result, dict)
