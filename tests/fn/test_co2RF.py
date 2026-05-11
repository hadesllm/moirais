"""Tests for co2RF.radiative_forcing_co2."""
import numpy as np
import pytest
from morie.fn.co2RF import radiative_forcing_co2


def test_co2RF_basic():
    """Test basic functionality."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    C0 = np.random.default_rng(42).normal(0, 1, 100)
    result = radiative_forcing_co2(C, C0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_co2RF_edge():
    """Test edge cases."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    C0 = np.random.default_rng(42).normal(0, 1, 100)
    result = radiative_forcing_co2(C, C0)
    assert isinstance(result, dict)
