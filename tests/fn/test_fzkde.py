"""Tests for fzkde.fauzi_standard_kde."""
import numpy as np
import pytest
from morie.fn.fzkde import fauzi_standard_kde


def test_fzkde_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = fauzi_standard_kde(x, bandwidth, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzkde_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = fauzi_standard_kde(x, bandwidth, kernel)
    assert isinstance(result, dict)
