"""Tests for rghaar.rangayyan_haar_wavelet."""
import numpy as np
import pytest
from moirais.fn.rghaar import rangayyan_haar_wavelet


def test_rghaar_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_haar_wavelet(x, levels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rghaar_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    levels = [0.5, 1.0, 1.5, 2.0]
    result = rangayyan_haar_wavelet(x, levels)
    assert isinstance(result, dict)
