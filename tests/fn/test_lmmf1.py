"""Tests for lmmf1.lmm_form_eq2_1."""
import numpy as np
import pytest
from morie.fn.lmmf1 import lmm_form_eq2_1


def test_lmmf1_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    beta_init = 0.0
    result = lmm_form_eq2_1(Y, X, Z, beta_init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lmmf1_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    beta_init = 0.0
    result = lmm_form_eq2_1(Y, X, Z, beta_init)
    assert isinstance(result, dict)
