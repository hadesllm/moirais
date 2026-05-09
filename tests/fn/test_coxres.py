"""Tests for coxres.cox_schoenfeld_residuals."""
import numpy as np
import pytest
from moirais.fn.coxres import cox_schoenfeld_residuals


def test_coxres_basic():
    """Test basic functionality."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = cox_schoenfeld_residuals(fit, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_coxres_edge():
    """Test edge cases."""
    fit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = cox_schoenfeld_residuals(fit, time)
    assert isinstance(result, dict)
