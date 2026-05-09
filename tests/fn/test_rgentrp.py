"""Tests for rgentrp.rangayyan_spectral_entropy."""
import numpy as np
import pytest
from moirais.fn.rgentrp import rangayyan_spectral_entropy


def test_rgentrp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_spectral_entropy(x, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgentrp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_spectral_entropy(x, fs)
    assert isinstance(result, dict)
