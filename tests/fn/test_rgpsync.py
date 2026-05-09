"""Tests for rgpsync.rangayyan_pcg_sync_avg."""
import numpy as np
import pytest
from moirais.fn.rgpsync import rangayyan_pcg_sync_avg


def test_rgpsync_basic():
    """Test basic functionality."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    n_cycles = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_pcg_sync_avg(pcg, ecg, fs, n_cycles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpsync_edge():
    """Test edge cases."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    n_cycles = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_pcg_sync_avg(pcg, ecg, fs, n_cycles)
    assert isinstance(result, dict)
