"""Tests for gb32vr.gibbons_runs_var."""
import numpy as np
import pytest
from morie.fn.gb32vr import gibbons_runs_var


def test_gb32vr_basic():
    """Test basic functionality."""
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_runs_var(n1, n2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb32vr_edge():
    """Test edge cases."""
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_runs_var(n1, n2)
    assert isinstance(result, dict)
