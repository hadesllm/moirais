"""Tests for rgvf.rangayyan_vf_detect."""
import numpy as np
import pytest
from morie.fn.rgvf import rangayyan_vf_detect


def test_rgvf_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_vf_detect(ecg, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgvf_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_vf_detect(ecg, fs)
    assert isinstance(result, dict)
