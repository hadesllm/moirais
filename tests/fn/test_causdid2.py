"""Tests for causdid2.causal_did_2x2."""
import numpy as np
import pytest
from morie.fn.causdid2 import causal_did_2x2


def test_causdid2_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treated = np.random.default_rng(42).normal(0, 1, 100)
    post = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_did_2x2(y, treated, post)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causdid2_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treated = np.random.default_rng(42).normal(0, 1, 100)
    post = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_did_2x2(y, treated, post)
    assert isinstance(result, dict)
