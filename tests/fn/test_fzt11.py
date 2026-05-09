"""Tests for fzt11.fauzi_thm1_1_bias_mgkde."""
import numpy as np
import pytest
from moirais.fn.fzt11 import fauzi_thm1_1_bias_mgkde


def test_fzt11_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_thm1_1_bias_mgkde(x, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt11_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    result = fauzi_thm1_1_bias_mgkde(x, bandwidth)
    assert isinstance(result, dict)
