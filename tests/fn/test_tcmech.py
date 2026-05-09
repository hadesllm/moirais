"""Tests for tcmech.truncated_cdp_mechanism."""
import numpy as np
import pytest
from moirais.fn.tcmech import truncated_cdp_mechanism


def test_tcmech_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    f_value = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = truncated_cdp_mechanism(y, f_value, C, epsilon, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tcmech_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    f_value = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = truncated_cdp_mechanism(y, f_value, C, epsilon, delta)
    assert isinstance(result, dict)
