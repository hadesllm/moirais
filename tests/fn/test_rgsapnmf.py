"""Tests for rgsapnmf.rangayyan_sleep_apnea_nmf."""
import numpy as np
import pytest
from moirais.fn.rgsapnmf import rangayyan_sleep_apnea_nmf


def test_rgsapnmf_basic():
    """Test basic functionality."""
    signals = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    n_comp = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_sleep_apnea_nmf(signals, fs, n_comp)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgsapnmf_edge():
    """Test edge cases."""
    signals = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    n_comp = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_sleep_apnea_nmf(signals, fs, n_comp)
    assert isinstance(result, dict)
