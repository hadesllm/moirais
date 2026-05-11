"""Tests for bayima.importance_sampling."""
import numpy as np
import pytest
from morie.fn.bayima import importance_sampling


def test_bayima_basic():
    """Test basic functionality."""
    target = np.random.default_rng(43).integers(0, 2, 100)
    proposal = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = importance_sampling(target, proposal, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayima_edge():
    """Test edge cases."""
    target = np.random.default_rng(43).integers(0, 2, 100)
    proposal = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = importance_sampling(target, proposal, n)
    assert isinstance(result, dict)
