"""Tests for gatV2.gat_v2."""
import numpy as np
import pytest
from moirais.fn.gatV2 import gat_v2


def test_gatV2_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = gat_v2(A, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gatV2_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = gat_v2(A, X)
    assert isinstance(result, dict)
