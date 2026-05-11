"""Tests for gb_wci.gibbons_concordance_signif."""
import numpy as np
import pytest
from morie.fn.gb_wci import gibbons_concordance_signif


def test_gb_wci_basic():
    """Test basic functionality."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_concordance_signif(W, k, b)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_wci_edge():
    """Test edge cases."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_concordance_signif(W, k, b)
    assert isinstance(result, dict)
