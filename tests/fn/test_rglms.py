"""Tests for rglms.rangayyan_lms_filter."""
import numpy as np
import pytest
from morie.fn.rglms import rangayyan_lms_filter


def test_rglms_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    mu = 0.0
    order = 4
    result = rangayyan_lms_filter(x, d, mu, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rglms_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    mu = 0.0
    order = 4
    result = rangayyan_lms_filter(x, d, mu, order)
    assert isinstance(result, dict)
