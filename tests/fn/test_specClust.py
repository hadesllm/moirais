"""Tests for specClust.spectral_clustering."""
import numpy as np
import pytest
from morie.fn.specClust import spectral_clustering


def test_specClust_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    k = 5
    result = spectral_clustering(A, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_specClust_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    k = 5
    result = spectral_clustering(A, k)
    assert isinstance(result, dict)
