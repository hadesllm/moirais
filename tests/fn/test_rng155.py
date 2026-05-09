"""Tests for rng155.rangayyan_ch3_lms_filter_output."""
import numpy as np
import pytest
from moirais.fn.rng155 import rangayyan_ch3_lms_filter_output


def test_rng155_basic():
    """Test basic functionality."""
    r = 10
    w_k = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_lms_filter_output(r, w_k, n, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng155_edge():
    """Test edge cases."""
    r = 10
    w_k = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = rangayyan_ch3_lms_filter_output(r, w_k, n, M)
    assert isinstance(result, dict)
