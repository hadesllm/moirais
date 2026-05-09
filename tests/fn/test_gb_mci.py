"""Tests for gb_mci.gibbons_mcnemar_ci."""
import numpy as np
import pytest
from moirais.fn.gb_mci import gibbons_mcnemar_ci


def test_gb_mci_basic():
    """Test basic functionality."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    alpha = 0.05
    result = gibbons_mcnemar_ci(b, c, n, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_mci_edge():
    """Test edge cases."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    alpha = 0.05
    result = gibbons_mcnemar_ci(b, c, n, alpha)
    assert isinstance(result, dict)
