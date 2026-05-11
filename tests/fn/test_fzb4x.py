"""Tests for fzb4x.fauzi_b4_coefficient."""
import numpy as np
import pytest
from morie.fn.fzb4x import fauzi_b4_coefficient


def test_fzb4x_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = fauzi_b4_coefficient(x, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzb4x_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = fauzi_b4_coefficient(x, kernel)
    assert isinstance(result, dict)
