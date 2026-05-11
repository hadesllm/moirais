"""Tests for rgsapn.rangayyan_sleep_apnea."""
import numpy as np
import pytest
from morie.fn.rgsapn import rangayyan_sleep_apnea


def test_rgsapn_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    spo2 = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_sleep_apnea(ecg, spo2, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgsapn_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    spo2 = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_sleep_apnea(ecg, spo2, fs)
    assert isinstance(result, dict)
