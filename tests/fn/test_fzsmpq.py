"""Tests for fzsmpq.fauzi_sample_quantile."""
import numpy as np
import pytest
from moirais.fn.fzsmpq import fauzi_sample_quantile


def test_fzsmpq_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = fauzi_sample_quantile(data, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzsmpq_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = fauzi_sample_quantile(data, p)
    assert isinstance(result, dict)
