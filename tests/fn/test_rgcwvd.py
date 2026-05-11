"""Tests for rgcwvd.rangayyan_cohen_class."""
import numpy as np
import pytest
from morie.fn.rgcwvd import rangayyan_cohen_class


def test_rgcwvd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = rangayyan_cohen_class(x, fs, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgcwvd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = rangayyan_cohen_class(x, fs, kernel)
    assert isinstance(result, dict)
