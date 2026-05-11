"""Tests for hrzkd2.horowitz_multivariate_kde."""
import numpy as np
import pytest
from morie.fn.hrzkd2 import horowitz_multivariate_kde


def test_hrzkd2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidths = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_multivariate_kde(x, bandwidths)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzkd2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidths = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_multivariate_kde(x, bandwidths)
    assert isinstance(result, dict)
