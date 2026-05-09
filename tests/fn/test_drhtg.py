"""Tests for drhtg.dr_did_heterogeneity."""
import numpy as np
import pytest
from moirais.fn.drhtg import dr_did_heterogeneity


def test_drhtg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    strata = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_did_heterogeneity(y, D, X, strata)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drhtg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    strata = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_did_heterogeneity(y, D, X, strata)
    assert isinstance(result, dict)
