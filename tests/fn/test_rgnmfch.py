"""Tests for rgnmfch.rangayyan_nmf_channel_sel."""
import numpy as np
import pytest
from morie.fn.rgnmfch import rangayyan_nmf_channel_sel


def test_rgnmfch_basic():
    """Test basic functionality."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    n_comp = np.random.default_rng(42).normal(0, 1, 100)
    n_select = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_nmf_channel_sel(eeg, n_comp, n_select)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgnmfch_edge():
    """Test edge cases."""
    eeg = np.random.default_rng(42).normal(0, 1, 1024)
    n_comp = np.random.default_rng(42).normal(0, 1, 100)
    n_select = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_nmf_channel_sel(eeg, n_comp, n_select)
    assert isinstance(result, dict)
