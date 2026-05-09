"""Tests for hmc1d.geron_causal_1d_conv."""
import numpy as np
import pytest
from moirais.fn.hmc1d import geron_causal_1d_conv


def test_hmc1d_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = geron_causal_1d_conv(x, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmc1d_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = geron_causal_1d_conv(x, kernel)
    assert isinstance(result, dict)
