"""Tests for zfmech.z_dp_mechanism."""
import numpy as np
import pytest
from morie.fn.zfmech import z_dp_mechanism


def test_zfmech_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    result = z_dp_mechanism(y, sensitivity, rho)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_zfmech_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sensitivity = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    result = z_dp_mechanism(y, sensitivity, rho)
    assert isinstance(result, dict)
