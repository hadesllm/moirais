"""Tests for gb_spv.gibbons_spearman_rho_var."""
import numpy as np
import pytest
from moirais.fn.gb_spv import gibbons_spearman_rho_var


def test_gb_spv_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_spearman_rho_var(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gb_spv_edge():
    """Test edge cases."""
    result = gibbons_spearman_rho_var(np.array([42.0]))
    assert result['n'] == 1
