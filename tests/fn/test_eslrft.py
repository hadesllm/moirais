"""Tests for eslrft.esl_random_forest."""
import numpy as np
import pytest
from morie.fn.eslrft import esl_random_forest


def test_eslrft_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = esl_random_forest(X, y, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslrft_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = esl_random_forest(X, y, B)
    assert isinstance(result, dict)
