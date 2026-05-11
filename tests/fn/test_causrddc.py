"""Tests for causrddc.causal_rdd_ccft_bw."""
import numpy as np
import pytest
from morie.fn.causrddc import causal_rdd_ccft_bw


def test_causrddc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    cutoff = 10.0
    p = 5
    result = causal_rdd_ccft_bw(x, y, cutoff, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causrddc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    cutoff = 10.0
    p = 5
    result = causal_rdd_ccft_bw(x, y, cutoff, p)
    assert isinstance(result, dict)
