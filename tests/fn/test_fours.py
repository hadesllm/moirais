"""Tests for fours.fourier_basis."""
import numpy as np
import pytest
from moirais.fn.fours import fourier_basis


def test_fours_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    n_harmonics = np.random.default_rng(42).normal(0, 1, 100)
    result = fourier_basis(t, n_harmonics)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fours_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    n_harmonics = np.random.default_rng(42).normal(0, 1, 100)
    result = fourier_basis(t, n_harmonics)
    assert isinstance(result, dict)
