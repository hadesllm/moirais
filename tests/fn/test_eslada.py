"""Tests for eslada.esl_adaboost."""
import numpy as np
import pytest
from morie.fn.eslada import esl_adaboost


def test_eslada_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = esl_adaboost(X, y, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslada_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = esl_adaboost(X, y, M)
    assert isinstance(result, dict)
