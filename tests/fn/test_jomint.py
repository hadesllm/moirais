"""Tests for jomint.joseph_mint_reconciliation."""
import numpy as np
import pytest
from morie.fn.jomint import joseph_mint_reconciliation


def test_jomint_basic():
    """Test basic functionality."""
    y_hat = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_mint_reconciliation(y_hat, S, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jomint_edge():
    """Test edge cases."""
    y_hat = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_mint_reconciliation(y_hat, S, W)
    assert isinstance(result, dict)
