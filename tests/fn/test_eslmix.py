"""Tests for eslmix.esl_gaussian_mixture."""
import numpy as np
import pytest
from moirais.fn.eslmix import esl_gaussian_mixture


def test_eslmix_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = esl_gaussian_mixture(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslmix_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = esl_gaussian_mixture(X, k)
    assert isinstance(result, dict)
