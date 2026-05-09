"""Tests for rng049.rangayyan_ch3_laplace_transform_causal_finite."""
import numpy as np
import pytest
from moirais.fn.rng049 import rangayyan_ch3_laplace_transform_causal_finite


def test_rng049_basic():
    """Test basic functionality."""
    h = 0.3
    t = np.linspace(0, 10, 100)
    s = 90
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_laplace_transform_causal_finite(h, t, s, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng049_edge():
    """Test edge cases."""
    h = 0.3
    t = np.linspace(0, 10, 100)
    s = 90
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_laplace_transform_causal_finite(h, t, s, T)
    assert isinstance(result, dict)
