"""Tests for gb1131.gibbons_spearman_rho."""
import numpy as np
import pytest
from morie.fn.gb1131 import gibbons_spearman_rho


def test_gb1131_basic():
    """Test basic functionality."""
    x = np.arange(10, dtype=float)
    y = x * 2 + 1
    result = gibbons_spearman_rho(x, y)
    assert abs(result['statistic'] - 1.0) < 0.01


def test_gb1131_edge():
    """Test edge cases."""
    result = gibbons_spearman_rho(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert result['n'] == 2
