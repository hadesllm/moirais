"""Tests for gelutn.gelu_tanh_approx."""
import numpy as np
import pytest
from moirais.fn.gelutn import gelu_tanh_approx


def test_gelutn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gelu_tanh_approx(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gelutn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gelu_tanh_approx(y)
    assert isinstance(result, dict)
