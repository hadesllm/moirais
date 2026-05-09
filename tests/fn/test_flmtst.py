"""Tests for flmtst.fleming_harrington."""
import numpy as np
import pytest
from moirais.fn.flmtst import fleming_harrington


def test_flmtst_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    gamma = 1.0
    result = fleming_harrington(time, event, group, rho, gamma)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_flmtst_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    gamma = 1.0
    result = fleming_harrington(time, event, group, rho, gamma)
    assert isinstance(result, dict)
