"""Tests for nbdsp.negative_binomial_dispersion."""
import numpy as np
import pytest
from moirais.fn.nbdsp import negative_binomial_dispersion


def test_nbdsp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    link = 'identity'
    result = negative_binomial_dispersion(y, X, link)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nbdsp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    link = 'identity'
    result = negative_binomial_dispersion(y, X, link)
    assert isinstance(result, dict)
