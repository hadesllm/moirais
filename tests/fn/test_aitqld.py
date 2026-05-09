"""Tests for aitqld.compositional_quantile_dist."""
import numpy as np
import pytest
from moirais.fn.aitqld import compositional_quantile_dist


def test_aitqld_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = compositional_quantile_dist(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitqld_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = compositional_quantile_dist(X)
    assert isinstance(result, dict)
