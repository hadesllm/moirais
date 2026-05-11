"""Tests for rng152.rangayyan_ch3_wiener_frequency_response_snr_form."""
import numpy as np
import pytest
from morie.fn.rng152 import rangayyan_ch3_wiener_frequency_response_snr_form


def test_rng152_basic():
    """Test basic functionality."""
    S_d = np.random.default_rng(42).normal(0, 1, 100)
    S_eta = np.random.default_rng(42).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_wiener_frequency_response_snr_form(S_d, S_eta, omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng152_edge():
    """Test edge cases."""
    S_d = np.random.default_rng(42).normal(0, 1, 100)
    S_eta = np.random.default_rng(42).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_wiener_frequency_response_snr_form(S_d, S_eta, omega)
    assert isinstance(result, dict)
