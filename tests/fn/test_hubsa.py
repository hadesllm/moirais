"""The art of doing mathematics consists in finding that special case which contains all the germs of generality. — David Hilbert"""
import numpy as np
import pytest
from moirais.fn.hubsa import hits_hubs_authorities


def test_hubsa_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    tol = 1e-6
    result = hits_hubs_authorities(y, A, tol)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hubsa_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    tol = 1e-6
    result = hits_hubs_authorities(y, A, tol)
    assert isinstance(result, dict)
