"""Tests for hmfmap.geron_feature_map."""
import numpy as np
import pytest
from moirais.fn.hmfmap import geron_feature_map


def test_hmfmap_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_feature_map(x, K, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmfmap_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_feature_map(x, K, b)
    assert isinstance(result, dict)
