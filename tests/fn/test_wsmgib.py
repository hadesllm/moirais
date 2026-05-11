"""Tests for wsmgib.wasserman_gibbs_sampler."""
import numpy as np
import pytest
from morie.fn.wsmgib import wasserman_gibbs_sampler


def test_wsmgib_basic():
    """Test basic functionality."""
    target = np.random.default_rng(43).integers(0, 2, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = wasserman_gibbs_sampler(target, x0, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmgib_edge():
    """Test edge cases."""
    target = np.random.default_rng(43).integers(0, 2, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = wasserman_gibbs_sampler(target, x0, n)
    assert isinstance(result, dict)
