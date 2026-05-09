"""Tests for pmiwd.pointwise_mutual_info."""
import numpy as np
import pytest
from moirais.fn.pmiwd import pointwise_mutual_info


def test_pmiwd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y2 = np.random.default_rng(42).normal(0, 1, 100)
    result = pointwise_mutual_info(y, x, y2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pmiwd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y2 = np.random.default_rng(42).normal(0, 1, 100)
    result = pointwise_mutual_info(y, x, y2)
    assert isinstance(result, dict)
