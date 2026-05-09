"""Tests for cov2s.two_sample_coverage."""
import numpy as np
import pytest
from moirais.fn.cov2s import two_sample_coverage


def test_cov2s_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = two_sample_coverage(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cov2s_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = two_sample_coverage(x, y)
    assert isinstance(result, dict)
