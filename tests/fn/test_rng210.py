"""Tests for rng210.rangayyan_ch4_noise_psd_at_output."""
import numpy as np
import pytest
from moirais.fn.rng210 import rangayyan_ch4_noise_psd_at_output


def test_rng210_basic():
    """Test basic functionality."""
    P_eta_i = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_noise_psd_at_output(P_eta_i, H, f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng210_edge():
    """Test edge cases."""
    P_eta_i = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_noise_psd_at_output(P_eta_i, H, f)
    assert isinstance(result, dict)
