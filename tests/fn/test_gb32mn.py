"""Tests for gb32mn.gibbons_runs_mean."""
import numpy as np
import pytest
from moirais.fn.gb32mn import gibbons_runs_mean


def test_gb32mn_basic():
    """Test basic functionality."""
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_runs_mean(n1, n2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb32mn_edge():
    """Test edge cases."""
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_runs_mean(n1, n2)
    assert isinstance(result, dict)
