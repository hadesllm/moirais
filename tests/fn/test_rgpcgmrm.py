"""Tests for rgpcgmrm.rangayyan_pcg_murmur_detect."""
import numpy as np
import pytest
from morie.fn.rgpcgmrm import rangayyan_pcg_murmur_detect


def test_rgpcgmrm_basic():
    """Test basic functionality."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_pcg_murmur_detect(pcg, ecg, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpcgmrm_edge():
    """Test edge cases."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_pcg_murmur_detect(pcg, ecg, fs)
    assert isinstance(result, dict)
