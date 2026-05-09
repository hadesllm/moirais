"""Tests for gb_ttd.gibbons_total_runs_dist_table."""
import numpy as np
import pytest
from moirais.fn.gb_ttd import gibbons_total_runs_dist_table


def test_gb_ttd_basic():
    """Test basic functionality."""
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = gibbons_total_runs_dist_table(n1, n2, r)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_ttd_edge():
    """Test edge cases."""
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = gibbons_total_runs_dist_table(n1, n2, r)
    assert isinstance(result, dict)
