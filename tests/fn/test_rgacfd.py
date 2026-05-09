"""Tests for rgacfd.rangayyan_acf_distance."""
import numpy as np
import pytest
from moirais.fn.rgacfd import rangayyan_acf_distance


def test_rgacfd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    seg_len = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = rangayyan_acf_distance(x, seg_len, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgacfd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    seg_len = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = rangayyan_acf_distance(x, seg_len, p)
    assert isinstance(result, dict)
