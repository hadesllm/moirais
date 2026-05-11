"""Tests for rgepiksv.rangayyan_epilepsy_ksvd."""
import numpy as np
import pytest
from morie.fn.rgepiksv import rangayyan_epilepsy_ksvd


def test_rgepiksv_basic():
    """Test basic functionality."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    dict_size = 100
    sparsity = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_epilepsy_ksvd(eeg, fs, dict_size, sparsity)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgepiksv_edge():
    """Test edge cases."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    dict_size = 100
    sparsity = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_epilepsy_ksvd(eeg, fs, dict_size, sparsity)
    assert isinstance(result, dict)
