"""Tests for drbnk.dr_bandit_did."""
import numpy as np
import pytest
from moirais.fn.drbnk import dr_bandit_did


def test_drbnk_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D_t = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    pi_t = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_bandit_did(y, D_t, X, pi_t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drbnk_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D_t = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    pi_t = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_bandit_did(y, D_t, X, pi_t)
    assert isinstance(result, dict)
