"""Tests for rglpca.rangayyan_lpc_analysis."""
import numpy as np
import pytest
from moirais.fn.rglpca import rangayyan_lpc_analysis


def test_rglpca_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    frame_len = np.random.default_rng(42).normal(0, 1, 100)
    hop_len = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_lpc_analysis(x, order, frame_len, hop_len, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rglpca_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    frame_len = np.random.default_rng(42).normal(0, 1, 100)
    hop_len = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_lpc_analysis(x, order, frame_len, hop_len, fs)
    assert isinstance(result, dict)
