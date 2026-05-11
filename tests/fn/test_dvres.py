"""Tests for dvres.deviance_residual_cox."""
import numpy as np
import pytest
from morie.fn.dvres import deviance_residual_cox


def test_dvres_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    fitted = np.random.default_rng(42).normal(0, 1, 100)
    result = deviance_residual_cox(time, event, fitted)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dvres_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    fitted = np.random.default_rng(42).normal(0, 1, 100)
    result = deviance_residual_cox(time, event, fitted)
    assert isinstance(result, dict)
