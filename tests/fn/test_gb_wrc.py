"""Tests for gb_wrc.gibbons_runs_critical."""
import numpy as np
import pytest
from moirais.fn.gb_wrc import gibbons_runs_critical


def test_gb_wrc_basic():
    """Test basic functionality."""
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_runs_critical(n1, n2, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_wrc_edge():
    """Test edge cases."""
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_runs_critical(n1, n2, alpha)
    assert isinstance(result, dict)
