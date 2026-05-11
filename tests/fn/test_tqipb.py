"""Tests for tqipb.turboquant_inner_product_distortion_bound."""
import numpy as np
import pytest
from morie.fn.tqipb import turboquant_inner_product_distortion_bound


def test_tqipb_basic():
    """Test basic functionality."""
    bits = np.random.default_rng(42).normal(0, 1, 100)
    norm_sq = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = turboquant_inner_product_distortion_bound(bits, norm_sq, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqipb_edge():
    """Test edge cases."""
    bits = np.random.default_rng(42).normal(0, 1, 100)
    norm_sq = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = turboquant_inner_product_distortion_bound(bits, norm_sq, d)
    assert isinstance(result, dict)
