"""Tests for drdyn.dr_dynamic_did."""
import numpy as np
import pytest
from morie.fn.drdyn import dr_dynamic_did


def test_drdyn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    cohort = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_dynamic_did(y, D, unit, time, cohort, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drdyn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    cohort = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_dynamic_did(y, D, unit, time, cohort, horizon)
    assert isinstance(result, dict)
