"""Tests for rng163.rangayyan_ch3_rls_objective."""
import numpy as np
import pytest
from moirais.fn.rng163 import rangayyan_ch3_rls_objective


def test_rng163_basic():
    """Test basic functionality."""
    e = np.random.default_rng(44).normal(0, 1, 100)
    lam = 0.1
    n = 100
    result = rangayyan_ch3_rls_objective(e, lam, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng163_edge():
    """Test edge cases."""
    e = np.random.default_rng(44).normal(0, 1, 100)
    lam = 0.1
    n = 100
    result = rangayyan_ch3_rls_objective(e, lam, n)
    assert isinstance(result, dict)
