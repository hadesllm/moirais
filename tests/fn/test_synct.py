"""Tests for synct.synthetic_control."""
import numpy as np
import pytest
from morie.fn.synct import synthetic_control


def test_synct_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    unit_id = np.random.default_rng(42).normal(0, 1, 100)
    time_id = np.random.default_rng(42).normal(0, 1, 100)
    treated_unit = np.random.default_rng(42).normal(0, 1, 100)
    treatment_time = np.random.default_rng(42).normal(0, 1, 100)
    result = synthetic_control(Y, unit_id, time_id, treated_unit, treatment_time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_synct_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    unit_id = np.random.default_rng(42).normal(0, 1, 100)
    time_id = np.random.default_rng(42).normal(0, 1, 100)
    treated_unit = np.random.default_rng(42).normal(0, 1, 100)
    treatment_time = np.random.default_rng(42).normal(0, 1, 100)
    result = synthetic_control(Y, unit_id, time_id, treated_unit, treatment_time)
    assert isinstance(result, dict)
