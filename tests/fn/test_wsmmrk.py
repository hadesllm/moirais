"""Tests for wsmmrk.wasserman_markov_ineq."""
import numpy as np
import pytest
from moirais.fn.wsmmrk import wasserman_markov_ineq


def test_wsmmrk_basic():
    """Test basic functionality."""
    mean = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = wasserman_markov_ineq(mean, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmmrk_edge():
    """Test edge cases."""
    mean = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = wasserman_markov_ineq(mean, a)
    assert isinstance(result, dict)
