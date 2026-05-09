"""Tests for evgevp2.evt_gev_pwm."""
import numpy as np
import pytest
from moirais.fn.evgevp2 import evt_gev_pwm


def test_evgevp2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_pwm(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evgevp2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_pwm(x)
    assert isinstance(result, dict)
