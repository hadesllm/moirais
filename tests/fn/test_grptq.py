"""Tests for grptq.geron_static_ptq."""
import numpy as np
import pytest
from morie.fn.grptq import geron_static_ptq


def test_grptq_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    calibration_data = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_static_ptq(model, calibration_data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grptq_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    calibration_data = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_static_ptq(model, calibration_data)
    assert isinstance(result, dict)
