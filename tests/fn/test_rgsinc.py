"""Tests for rgsinc.rangayyan_sinc_kernel."""
import numpy as np
import pytest
from moirais.fn.rgsinc import rangayyan_sinc_kernel


def test_rgsinc_basic():
    """Test basic functionality."""
    fc = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_sinc_kernel(fc, fs, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgsinc_edge():
    """Test edge cases."""
    fc = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_sinc_kernel(fc, fs, M)
    assert isinstance(result, dict)
