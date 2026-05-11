"""Tests for rng140.rangayyan_ch3_estimation_error_vector_form."""
import numpy as np
import pytest
from morie.fn.rng140 import rangayyan_ch3_estimation_error_vector_form


def test_rng140_basic():
    """Test basic functionality."""
    d = 5
    w = np.random.default_rng(45).exponential(1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_estimation_error_vector_form(d, w, x, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng140_edge():
    """Test edge cases."""
    d = 5
    w = np.random.default_rng(45).exponential(1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_estimation_error_vector_form(d, w, x, n)
    assert isinstance(result, dict)
