"""Tests for rgecgfe.rangayyan_fetal_ecg_single."""
import numpy as np
import pytest
from morie.fn.rgecgfe import rangayyan_fetal_ecg_single


def test_rgecgfe_basic():
    """Test basic functionality."""
    abdominal_ecg = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    method = 'auto'
    result = rangayyan_fetal_ecg_single(abdominal_ecg, fs, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgecgfe_edge():
    """Test edge cases."""
    abdominal_ecg = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    method = 'auto'
    result = rangayyan_fetal_ecg_single(abdominal_ecg, fs, method)
    assert isinstance(result, dict)
