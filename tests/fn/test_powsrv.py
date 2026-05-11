"""Tests for powsrv.power_survey."""
import numpy as np
import pytest
from morie.fn.powsrv import power_survey


def test_powsrv_basic():
    """Test basic functionality."""
    effect_size = 100
    alpha = 0.05
    DEFF = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = power_survey(effect_size, alpha, DEFF, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_powsrv_edge():
    """Test edge cases."""
    effect_size = 100
    alpha = 0.05
    DEFF = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = power_survey(effect_size, alpha, DEFF, n)
    assert isinstance(result, dict)
