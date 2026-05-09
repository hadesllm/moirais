"""Tests for gibbsm.gibbs_sampler."""
import numpy as np
import pytest
from moirais.fn.gibbsm import gibbs_sampler


def test_gibbsm_basic():
    """Test basic functionality."""
    conditionals = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = gibbs_sampler(conditionals, x0, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gibbsm_edge():
    """Test edge cases."""
    conditionals = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = gibbs_sampler(conditionals, x0, n_iter)
    assert isinstance(result, dict)
