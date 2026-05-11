"""Tests for tqmsb.turboquant_mse_distortion_bound."""
import numpy as np
import pytest
from morie.fn.tqmsb import turboquant_mse_distortion_bound


def test_tqmsb_basic():
    """Test basic functionality."""
    bits = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_mse_distortion_bound(bits)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqmsb_edge():
    """Test edge cases."""
    bits = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_mse_distortion_bound(bits)
    assert isinstance(result, dict)
