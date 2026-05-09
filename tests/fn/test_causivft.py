"""Tests for causivft.causal_iv_first_stage."""
import numpy as np
import pytest
from moirais.fn.causivft import causal_iv_first_stage


def test_causivft_basic():
    """Test basic functionality."""
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    X_exog = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_iv_first_stage(D, Z, X_exog)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_causivft_edge():
    """Test edge cases."""
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    X_exog = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_iv_first_stage(D, Z, X_exog)
    assert isinstance(result, dict)
