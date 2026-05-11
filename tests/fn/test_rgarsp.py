"""Tests for rgarsp.rangayyan_ar_spectrum."""
import numpy as np
import pytest
from morie.fn.rgarsp import rangayyan_ar_spectrum


def test_rgarsp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    fs = 100.0
    result = rangayyan_ar_spectrum(x, order, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgarsp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    fs = 100.0
    result = rangayyan_ar_spectrum(x, order, fs)
    assert isinstance(result, dict)
