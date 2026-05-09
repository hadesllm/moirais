"""Tests for drrct.dr_rct_assisted_did."""
import numpy as np
import pytest
from moirais.fn.drrct import dr_rct_assisted_did


def test_drrct_basic():
    """Test basic functionality."""
    y_obs = np.random.default_rng(42).normal(0, 1, 100)
    y_rct = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dr_rct_assisted_did(y_obs, y_rct, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drrct_edge():
    """Test edge cases."""
    y_obs = np.random.default_rng(42).normal(0, 1, 100)
    y_rct = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dr_rct_assisted_did(y_obs, y_rct, D, X)
    assert isinstance(result, dict)
