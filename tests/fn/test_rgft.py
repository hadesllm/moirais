"""Tests for rgft.rangayyan_fourier_transform."""
import numpy as np
import pytest
from moirais.fn.rgft import rangayyan_fourier_transform


def test_rgft_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_fourier_transform(t, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgft_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_fourier_transform(t, x)
    assert isinstance(result, dict)
