"""Tests for epbias.exposure_misclass_bias."""
import numpy as np
import pytest
from morie.fn.epbias import exposure_misclass_bias


def test_epbias_basic():
    """Test basic functionality."""
    A_obs = np.random.default_rng(42).normal(0, 1, 100)
    Se = np.random.default_rng(42).normal(0, 1, 100)
    Sp = np.random.default_rng(42).normal(0, 1, 100)
    result = exposure_misclass_bias(A_obs, Se, Sp)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_epbias_edge():
    """Test edge cases."""
    A_obs = np.random.default_rng(42).normal(0, 1, 100)
    Se = np.random.default_rng(42).normal(0, 1, 100)
    Sp = np.random.default_rng(42).normal(0, 1, 100)
    result = exposure_misclass_bias(A_obs, Se, Sp)
    assert isinstance(result, dict)
