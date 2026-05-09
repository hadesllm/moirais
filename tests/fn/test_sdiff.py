"""Tests for sdiff.synthetic_did."""
import numpy as np
import pytest
from moirais.fn.sdiff import synthetic_did


def test_sdiff_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    unit_id = np.random.default_rng(42).normal(0, 1, 100)
    time_id = np.random.default_rng(42).normal(0, 1, 100)
    treated = np.random.default_rng(42).normal(0, 1, 100)
    treatment_time = np.random.default_rng(42).normal(0, 1, 100)
    result = synthetic_did(Y, unit_id, time_id, treated, treatment_time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sdiff_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    unit_id = np.random.default_rng(42).normal(0, 1, 100)
    time_id = np.random.default_rng(42).normal(0, 1, 100)
    treated = np.random.default_rng(42).normal(0, 1, 100)
    treatment_time = np.random.default_rng(42).normal(0, 1, 100)
    result = synthetic_did(Y, unit_id, time_id, treated, treatment_time)
    assert isinstance(result, dict)
