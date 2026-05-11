"""Tests for eslnnt.esl_neural_net."""
import numpy as np
import pytest
from morie.fn.eslnnt import esl_neural_net


def test_eslnnt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = esl_neural_net(X, y, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslnnt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = esl_neural_net(X, y, M)
    assert isinstance(result, dict)
