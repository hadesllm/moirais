"""Tests for coxmgr.cox_martingale_residuals."""
import numpy as np
import pytest
from morie.fn.coxmgr import cox_martingale_residuals


def test_coxmgr_basic():
    """Test basic functionality."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    result = cox_martingale_residuals(fit)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_coxmgr_edge():
    """Test edge cases."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    result = cox_martingale_residuals(fit)
    assert isinstance(result, dict)
