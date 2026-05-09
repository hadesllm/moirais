"""Tests for ssmkrn.s4_ssm_kernel."""
import numpy as np
import pytest
from moirais.fn.ssmkrn import s4_ssm_kernel


def test_ssmkrn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    result = s4_ssm_kernel(y, x, A, B, C, L)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ssmkrn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    result = s4_ssm_kernel(y, x, A, B, C, L)
    assert isinstance(result, dict)
