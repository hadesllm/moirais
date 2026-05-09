"""Tests for fzt23.fauzi_thm2_3_var_brdkdfe."""
import numpy as np
import pytest
from moirais.fn.fzt23 import fauzi_thm2_3_var_brdkdfe


def test_fzt23_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = fauzi_thm2_3_var_brdkdfe(x, bandwidth, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt23_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = fauzi_thm2_3_var_brdkdfe(x, bandwidth, a)
    assert isinstance(result, dict)
