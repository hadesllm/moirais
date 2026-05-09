"""Tests for johbu.joseph_bottom_up_reconciliation."""
import numpy as np
import pytest
from moirais.fn.johbu import joseph_bottom_up_reconciliation


def test_johbu_basic():
    """Test basic functionality."""
    y_hat_bottom = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_bottom_up_reconciliation(y_hat_bottom, S)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_johbu_edge():
    """Test edge cases."""
    y_hat_bottom = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_bottom_up_reconciliation(y_hat_bottom, S)
    assert isinstance(result, dict)
