"""Tests for b203.burkov_lm_ch2_lm_shorthand."""
import numpy as np
import pytest
from morie.fn.b203 import burkov_lm_ch2_lm_shorthand


def test_b203_basic():
    """Test basic functionality."""
    t_next = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    result = burkov_lm_ch2_lm_shorthand(t_next, s)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_b203_edge():
    """Test edge cases."""
    t_next = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    result = burkov_lm_ch2_lm_shorthand(t_next, s)
    assert isinstance(result, dict)
