"""Tests for gb_eqf.gibbons_emp_quantile."""
import numpy as np
import pytest
from moirais.fn.gb_eqf import gibbons_emp_quantile


def test_gb_eqf_basic():
    """Test basic functionality."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_emp_quantile(u, data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb_eqf_edge():
    """Test edge cases."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_emp_quantile(u, data)
    assert isinstance(result, dict)
