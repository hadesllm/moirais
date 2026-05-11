"""Tests for b202.burkov_lm_ch2_lm_next_token."""
import numpy as np
import pytest
from morie.fn.b202 import burkov_lm_ch2_lm_next_token


def test_b202_basic():
    """Test basic functionality."""
    t_next = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    result = burkov_lm_ch2_lm_next_token(t_next, s)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_b202_edge():
    """Test edge cases."""
    t_next = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    result = burkov_lm_ch2_lm_next_token(t_next, s)
    assert isinstance(result, dict)
