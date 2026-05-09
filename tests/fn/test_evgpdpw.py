"""Tests for evgpdpw.evt_gpd_pwm."""
import numpy as np
import pytest
from moirais.fn.evgpdpw import evt_gpd_pwm


def test_evgpdpw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = evt_gpd_pwm(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evgpdpw_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = evt_gpd_pwm(y)
    assert isinstance(result, dict)
