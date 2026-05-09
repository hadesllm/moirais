"""Tests for bayscm.bayes_c_pi."""
import numpy as np
import pytest
from moirais.fn.bayscm import bayes_c_pi


def test_bayscm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    pi = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_c_pi(y, M, pi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayscm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    pi = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_c_pi(y, M, pi)
    assert isinstance(result, dict)
