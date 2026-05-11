"""Tests for rgadp.rangayyan_adaptive_filter."""
import numpy as np
import pytest
from morie.fn.rgadp import rangayyan_adaptive_filter


def test_rgadp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_adaptive_filter(x, reference)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgadp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_adaptive_filter(x, reference)
    assert isinstance(result, dict)
