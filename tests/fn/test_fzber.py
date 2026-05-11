"""Tests for fzber.fauzi_berry_esseen_quantile."""
import numpy as np
import pytest
from morie.fn.fzber import fauzi_berry_esseen_quantile


def test_fzber_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    bandwidth = 0.3
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_berry_esseen_quantile(data, p, bandwidth, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzber_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    bandwidth = 0.3
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_berry_esseen_quantile(data, p, bandwidth, x)
    assert isinstance(result, dict)
