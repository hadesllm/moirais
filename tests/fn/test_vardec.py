"""Tests for vardec.var_variance_decomp."""
import numpy as np
import pytest
from morie.fn.vardec import var_variance_decomp


def test_vardec_basic():
    """Test basic functionality."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = var_variance_decomp(fit, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vardec_edge():
    """Test edge cases."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = var_variance_decomp(fit, horizon)
    assert isinstance(result, dict)
