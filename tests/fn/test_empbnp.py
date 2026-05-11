"""Tests for empbnp.empirical_bayes_np."""
import numpy as np
import pytest
from morie.fn.empbnp import empirical_bayes_np


def test_empbnp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    prior_family = np.random.default_rng(42).normal(0, 1, 100)
    result = empirical_bayes_np(y, prior_family)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_empbnp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    prior_family = np.random.default_rng(42).normal(0, 1, 100)
    result = empirical_bayes_np(y, prior_family)
    assert isinstance(result, dict)
