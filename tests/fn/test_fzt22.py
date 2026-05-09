"""Tests for fzt22.fauzi_thm2_2_bias_brdkdfe."""
import numpy as np
import pytest
from moirais.fn.fzt22 import fauzi_thm2_2_bias_brdkdfe


def test_fzt22_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = fauzi_thm2_2_bias_brdkdfe(x, bandwidth, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt22_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = fauzi_thm2_2_bias_brdkdfe(x, bandwidth, a)
    assert isinstance(result, dict)
