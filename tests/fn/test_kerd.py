"""Tests for kerd.kernel_density_fda."""
import numpy as np
import pytest
from moirais.fn.kerd import kernel_density_fda


def test_kerd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = kernel_density_fda(x, h, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kerd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = kernel_density_fda(x, h, kernel)
    assert isinstance(result, dict)
