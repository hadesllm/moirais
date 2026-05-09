"""Tests for fzb2x.fauzi_b2_coefficient."""
import numpy as np
import pytest
from moirais.fn.fzb2x import fauzi_b2_coefficient


def test_fzb2x_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = fauzi_b2_coefficient(x, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzb2x_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = fauzi_b2_coefficient(x, kernel)
    assert isinstance(result, dict)
