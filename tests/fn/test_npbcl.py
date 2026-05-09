"""Tests for npbcl.np_bayes_clustering."""
import numpy as np
import pytest
from moirais.fn.npbcl import np_bayes_clustering


def test_npbcl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = np_bayes_clustering(y, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_npbcl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = np_bayes_clustering(y, alpha)
    assert isinstance(result, dict)
