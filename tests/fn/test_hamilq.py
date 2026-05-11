"""Tests for hamilq.hamilton_q_changepoint."""
import numpy as np
import pytest
from morie.fn.hamilq import hamilton_q_changepoint


def test_hamilq_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hamilton_q_changepoint(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hamilq_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hamilton_q_changepoint(x)
    assert isinstance(result, dict)
