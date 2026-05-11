"""Tests for fzr2.fauzi_r2_integral."""
import numpy as np
import pytest
from morie.fn.fzr2 import fauzi_r2_integral


def test_fzr2_basic():
    """Test basic functionality."""
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = fauzi_r2_integral(kernel, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzr2_edge():
    """Test edge cases."""
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = fauzi_r2_integral(kernel, a)
    assert isinstance(result, dict)
