"""Tests for rgmfcc.rangayyan_mfcc."""
import numpy as np
import pytest
from morie.fn.rgmfcc import rangayyan_mfcc


def test_rgmfcc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    n_mfcc = np.random.default_rng(42).normal(0, 1, 100)
    n_filters = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_mfcc(x, fs, n_mfcc, n_filters)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgmfcc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    n_mfcc = np.random.default_rng(42).normal(0, 1, 100)
    n_filters = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_mfcc(x, fs, n_mfcc, n_filters)
    assert isinstance(result, dict)
