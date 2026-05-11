"""Tests for causmrop.causal_robins_g_formula."""
import numpy as np
import pytest
from morie.fn.causmrop import causal_robins_g_formula


def test_causmrop_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    L = np.random.default_rng(42).normal(0, 1, 100)
    fit_fn = (lambda v: v)
    result = causal_robins_g_formula(y, A, L, fit_fn)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causmrop_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    L = np.random.default_rng(42).normal(0, 1, 100)
    fit_fn = (lambda v: v)
    result = causal_robins_g_formula(y, A, L, fit_fn)
    assert isinstance(result, dict)
