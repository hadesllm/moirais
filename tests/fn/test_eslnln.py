"""Tests for eslnln.esl_elastic_net."""
import numpy as np
import pytest
from morie.fn.eslnln import esl_elastic_net


def test_eslnln_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = esl_elastic_net(X, y, lambda_, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslnln_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = esl_elastic_net(X, y, lambda_, alpha)
    assert isinstance(result, dict)
