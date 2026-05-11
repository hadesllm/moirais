"""Tests for mcmpp.mcmcpack_irt."""
import numpy as np
import pytest
from morie.fn.mcmpp import mcmcpack_irt


def test_mcmpp_basic():
    """Test basic functionality."""
    votes = np.random.default_rng(43).integers(0, 2, (50, 100))
    n_dims = 2
    burnin = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = mcmcpack_irt(votes, n_dims, burnin, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mcmpp_edge():
    """Test edge cases."""
    votes = np.random.default_rng(43).integers(0, 2, (50, 100))
    n_dims = 2
    burnin = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = mcmcpack_irt(votes, n_dims, burnin, n_iter)
    assert isinstance(result, dict)
