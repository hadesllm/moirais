"""Tests for dprcl.dp_release_calibration."""
import numpy as np
import pytest
from moirais.fn.dprcl import dp_release_calibration


def test_dprcl_basic():
    """Test basic functionality."""
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_release_calibration(sensitivity, epsilon, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dprcl_edge():
    """Test edge cases."""
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_release_calibration(sensitivity, epsilon, c)
    assert isinstance(result, dict)
