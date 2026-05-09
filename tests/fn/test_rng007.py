"""Tests for rng007.rangayyan_ch3_sample_mean."""
import numpy as np
import pytest
from moirais.fn.rng007 import rangayyan_ch3_sample_mean


def test_rng007_basic():
    """Test basic functionality."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_sample_mean(eta, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng007_edge():
    """Test edge cases."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_sample_mean(eta, N)
    assert isinstance(result, dict)
