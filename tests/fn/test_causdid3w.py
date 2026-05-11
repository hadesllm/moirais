"""Tests for causdid3w.causal_did_three_way."""
import numpy as np
import pytest
from morie.fn.causdid3w import causal_did_three_way


def test_causdid3w_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treated = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = causal_did_three_way(y, treated, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causdid3w_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treated = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = causal_did_three_way(y, treated, t)
    assert isinstance(result, dict)
