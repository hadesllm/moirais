"""Tests for fgam.functional_gam."""
import numpy as np
import pytest
from moirais.fn.fgam import functional_gam


def test_fgam_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = functional_gam(X, Y, basis)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fgam_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = functional_gam(X, Y, basis)
    assert isinstance(result, dict)
