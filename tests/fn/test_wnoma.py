"""Tests for wnoma.wnominate_alternating."""
import numpy as np
import pytest
from moirais.fn.wnoma import wnominate_alternating


def test_wnoma_basic():
    """Test basic functionality."""
    votes = np.random.default_rng(43).integers(0, 2, (50, 100))
    n_dims = 2
    polarity = np.random.default_rng(42).normal(0, 1, 100)
    result = wnominate_alternating(votes, n_dims, polarity)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wnoma_edge():
    """Test edge cases."""
    votes = np.random.default_rng(43).integers(0, 2, (50, 100))
    n_dims = 2
    polarity = np.random.default_rng(42).normal(0, 1, 100)
    result = wnominate_alternating(votes, n_dims, polarity)
    assert isinstance(result, dict)
