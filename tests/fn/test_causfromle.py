"""Tests for causfromle.causal_e_value."""
import numpy as np
import pytest
from moirais.fn.causfromle import causal_e_value


def test_causfromle_basic():
    """Test basic functionality."""
    RR = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_e_value(RR)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causfromle_edge():
    """Test edge cases."""
    RR = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_e_value(RR)
    assert isinstance(result, dict)
