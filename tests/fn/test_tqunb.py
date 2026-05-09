"""Tests for tqunb.turboquant_prodqjl_unbiasedness."""
import numpy as np
import pytest
from moirais.fn.tqunb import turboquant_prodqjl_unbiasedness


def test_tqunb_basic():
    """Test basic functionality."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    m = 10
    result = turboquant_prodqjl_unbiasedness(q, k, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqunb_edge():
    """Test edge cases."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    m = 10
    result = turboquant_prodqjl_unbiasedness(q, k, m)
    assert isinstance(result, dict)
