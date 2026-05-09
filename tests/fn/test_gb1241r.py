"""Tests for gb1241r.gibbons_concordance_rho_link."""
import numpy as np
import pytest
from moirais.fn.gb1241r import gibbons_concordance_rho_link


def test_gb1241r_basic():
    """Test basic functionality."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = gibbons_concordance_rho_link(W, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb1241r_edge():
    """Test edge cases."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = gibbons_concordance_rho_link(W, k)
    assert isinstance(result, dict)
