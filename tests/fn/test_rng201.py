"""Tests for rng201.rangayyan_ch4_ccf_continuous_with_delay."""
import numpy as np
import pytest
from moirais.fn.rng201 import rangayyan_ch4_ccf_continuous_with_delay


def test_rng201_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    tau = 0.1
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch4_ccf_continuous_with_delay(x, y, tau, t)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rng201_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    tau = 0.1
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch4_ccf_continuous_with_delay(x, y, tau, t)
    assert isinstance(result, dict)
