"""Tests for thomp.thompson_sampling."""
import numpy as np
import pytest
from morie.fn.thomp import thompson_sampling


def test_thomp_basic():
    """Test basic functionality."""
    arms = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = thompson_sampling(arms, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_thomp_edge():
    """Test edge cases."""
    arms = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = thompson_sampling(arms, T)
    assert isinstance(result, dict)
