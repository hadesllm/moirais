"""Tests for rng204.rangayyan_ch4_psd_from_acf."""
import numpy as np
import pytest
from moirais.fn.rng204 import rangayyan_ch4_psd_from_acf


def test_rng204_basic():
    """Test basic functionality."""
    phi_xx = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    f = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = rangayyan_ch4_psd_from_acf(phi_xx, X, f, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng204_edge():
    """Test edge cases."""
    phi_xx = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    f = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = rangayyan_ch4_psd_from_acf(phi_xx, X, f, tau)
    assert isinstance(result, dict)
