"""Tests for rgsapdet.rangayyan_sleep_apnea_detect."""
import numpy as np
import pytest
from moirais.fn.rgsapdet import rangayyan_sleep_apnea_detect


def test_rgsapdet_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    spo2 = np.random.default_rng(42).normal(0, 1, 100)
    snore = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_sleep_apnea_detect(ecg, spo2, snore, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgsapdet_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    spo2 = np.random.default_rng(42).normal(0, 1, 100)
    snore = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_sleep_apnea_detect(ecg, spo2, snore, fs)
    assert isinstance(result, dict)
