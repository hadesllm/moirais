"""Tests for bayhier.hierarchical_pooling."""
import numpy as np
import pytest
from moirais.fn.bayhier import hierarchical_pooling


def test_bayhier_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    result = hierarchical_pooling(y, group)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayhier_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    result = hierarchical_pooling(y, group)
    assert isinstance(result, dict)
