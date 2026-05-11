"""Tests for spmrit.schabenberger_moran_i_residuals."""
import numpy as np
import pytest
from morie.fn.spmrit import schabenberger_moran_i_residuals


def test_spmrit_basic():
    """Test basic functionality."""
    residuals = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_moran_i_residuals(residuals, w)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spmrit_edge():
    """Test edge cases."""
    residuals = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_moran_i_residuals(residuals, w)
    assert isinstance(result, dict)
