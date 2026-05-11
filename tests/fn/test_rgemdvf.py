"""Tests for rgemdvf.rangayyan_emd_vf_detect."""
import numpy as np
import pytest
from morie.fn.rgemdvf import rangayyan_emd_vf_detect


def test_rgemdvf_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    n_imfs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_emd_vf_detect(ecg, fs, n_imfs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgemdvf_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    n_imfs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_emd_vf_detect(ecg, fs, n_imfs)
    assert isinstance(result, dict)
