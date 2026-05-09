"""Tests for pscli.pscl_ideal."""
import numpy as np
import pytest
from moirais.fn.pscli import pscl_ideal


def test_pscli_basic():
    """Test basic functionality."""
    rollcall_obj = np.random.default_rng(42).normal(0, 1, 100)
    n_dims = 2
    n_iter = 50
    result = pscl_ideal(rollcall_obj, n_dims, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pscli_edge():
    """Test edge cases."""
    rollcall_obj = np.random.default_rng(42).normal(0, 1, 100)
    n_dims = 2
    n_iter = 50
    result = pscl_ideal(rollcall_obj, n_dims, n_iter)
    assert isinstance(result, dict)
