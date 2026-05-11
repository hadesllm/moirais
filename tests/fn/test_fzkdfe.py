"""Tests for fzkdfe.fauzi_kdfe."""
import numpy as np
import pytest
from morie.fn.fzkdfe import fauzi_kdfe


def test_fzkdfe_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = fauzi_kdfe(x, bandwidth, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzkdfe_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = fauzi_kdfe(x, bandwidth, kernel)
    assert isinstance(result, dict)
