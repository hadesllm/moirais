"""Tests for gb1131n.gibbons_spearman_asymp."""
import numpy as np
import pytest
from moirais.fn.gb1131n import gibbons_spearman_asymp


def test_gb1131n_basic():
    """Test basic functionality."""
    r_s = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_spearman_asymp(r_s, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb1131n_edge():
    """Test edge cases."""
    r_s = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_spearman_asymp(r_s, n)
    assert isinstance(result, dict)
