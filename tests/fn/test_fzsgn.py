"""Tests for fzsgn.fauzi_smoothed_sign."""
import numpy as np
import pytest
from moirais.fn.fzsgn import fauzi_smoothed_sign


def test_fzsgn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_smoothed_sign(x)
    assert 'statistic' in result
    assert 'p_value' in result
    assert 0 <= result['p_value'] <= 1


def test_fzsgn_edge():
    """Test edge cases."""
    result = fauzi_smoothed_sign(np.array([1.0]))
    assert result['n'] == 1
