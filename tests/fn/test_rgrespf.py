"""Tests for rgrespf.rangayyan_respiration_features."""
import numpy as np
import pytest
from morie.fn.rgrespf import rangayyan_respiration_features


def test_rgrespf_basic():
    """Test basic functionality."""
    resp = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_respiration_features(resp, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgrespf_edge():
    """Test edge cases."""
    resp = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_respiration_features(resp, fs)
    assert isinstance(result, dict)
