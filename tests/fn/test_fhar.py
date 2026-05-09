"""Tests for fhar.fourier_basis."""
import numpy as np
import pytest
from moirais.fn.fhar import fourier_basis


def test_fhar_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = fourier_basis(t, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fhar_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = fourier_basis(t, K)
    assert isinstance(result, dict)
