"""Tests for rng174.rangayyan_ch3_rls_weight_update_compact."""
import numpy as np
import pytest
from morie.fn.rng174 import rangayyan_ch3_rls_weight_update_compact


def test_rng174_basic():
    """Test basic functionality."""
    w_tilde = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    alpha = 0.05
    n = 100
    result = rangayyan_ch3_rls_weight_update_compact(w_tilde, k, alpha, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng174_edge():
    """Test edge cases."""
    w_tilde = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    alpha = 0.05
    n = 100
    result = rangayyan_ch3_rls_weight_update_compact(w_tilde, k, alpha, n)
    assert isinstance(result, dict)
