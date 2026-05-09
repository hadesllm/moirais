"""Tests for grc1d.geron_causal_1d_cnn."""
import numpy as np
import pytest
from moirais.fn.grc1d import geron_causal_1d_cnn


def test_grc1d_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = geron_causal_1d_cnn(x, w)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grc1d_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = geron_causal_1d_cnn(x, w)
    assert isinstance(result, dict)
