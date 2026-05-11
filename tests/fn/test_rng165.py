"""Tests for rng165.rangayyan_ch3_rls_phi_matrix."""
import numpy as np
import pytest
from morie.fn.rng165 import rangayyan_ch3_rls_phi_matrix


def test_rng165_basic():
    """Test basic functionality."""
    r = 10
    lam = 0.1
    n = 100
    result = rangayyan_ch3_rls_phi_matrix(r, lam, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng165_edge():
    """Test edge cases."""
    r = 10
    lam = 0.1
    n = 100
    result = rangayyan_ch3_rls_phi_matrix(r, lam, n)
    assert isinstance(result, dict)
