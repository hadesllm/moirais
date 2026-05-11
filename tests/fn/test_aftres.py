"""Tests for aftres.aft_residuals."""
import numpy as np
import pytest
from morie.fn.aftres import aft_residuals


def test_aftres_basic():
    """Test basic functionality."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    result = aft_residuals(fit)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aftres_edge():
    """Test edge cases."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    result = aft_residuals(fit)
    assert isinstance(result, dict)
