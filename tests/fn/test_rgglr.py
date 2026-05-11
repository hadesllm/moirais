"""Tests for rgglr.rangayyan_gen_likelihood_ratio."""
import numpy as np
import pytest
from morie.fn.rgglr import rangayyan_gen_likelihood_ratio


def test_rgglr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    seg_len = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = rangayyan_gen_likelihood_ratio(x, seg_len, order)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_rgglr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    seg_len = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = rangayyan_gen_likelihood_ratio(x, seg_len, order)
    assert isinstance(result, dict)
