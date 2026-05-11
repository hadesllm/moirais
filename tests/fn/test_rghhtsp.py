"""Tests for rghhtsp.rangayyan_hht_spectrum."""
import numpy as np
import pytest
from morie.fn.rghhtsp import rangayyan_hht_spectrum


def test_rghhtsp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    max_imfs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_hht_spectrum(x, fs, max_imfs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rghhtsp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    max_imfs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_hht_spectrum(x, fs, max_imfs)
    assert isinstance(result, dict)
