"""Tests for fzksm.fauzi_ks_smoothed."""
import numpy as np
import pytest
from moirais.fn.fzksm import fauzi_ks_smoothed


def test_fzksm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_ks_smoothed(x)
    assert 'statistic' in result
    assert 'p_value' in result
    assert 0 <= result['p_value'] <= 1


def test_fzksm_edge():
    """Test edge cases."""
    result = fauzi_ks_smoothed(np.array([1.0]))
    assert result['n'] == 1
