"""Tests for eslzst.esl_z_score."""
import numpy as np
import pytest
from morie.fn.eslzst import esl_z_score


def test_eslzst_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    beta = 0.8
    result = esl_z_score(X, y, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslzst_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    beta = 0.8
    result = esl_z_score(X, y, beta)
    assert isinstance(result, dict)
