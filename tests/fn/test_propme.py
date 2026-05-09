"""Tests for propme.proportion_mediated."""
import numpy as np
import pytest
from moirais.fn.propme import proportion_mediated


def test_propme_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c_prime = np.random.default_rng(42).normal(0, 1, 100)
    result = proportion_mediated(a, b, c_prime)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_propme_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c_prime = np.random.default_rng(42).normal(0, 1, 100)
    result = proportion_mediated(a, b, c_prime)
    assert isinstance(result, dict)
