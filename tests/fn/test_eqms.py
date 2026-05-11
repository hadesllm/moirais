"""Tests for eqms.equating_mean_sigma."""
import numpy as np
import pytest
from morie.fn.eqms import equating_mean_sigma


def test_eqms_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    b_R = np.random.default_rng(42).normal(0, 1, 100)
    b_F = np.random.default_rng(42).normal(0, 1, 100)
    result = equating_mean_sigma(y, b_R, b_F)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eqms_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    b_R = np.random.default_rng(42).normal(0, 1, 100)
    b_F = np.random.default_rng(42).normal(0, 1, 100)
    result = equating_mean_sigma(y, b_R, b_F)
    assert isinstance(result, dict)
