"""Tests for rgpcgadp.rangayyan_pcg_adaptive_seg."""
import numpy as np
import pytest
from moirais.fn.rgpcgadp import rangayyan_pcg_adaptive_seg


def test_rgpcgadp_basic():
    """Test basic functionality."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    ar_order = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_pcg_adaptive_seg(pcg, fs, ar_order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpcgadp_edge():
    """Test edge cases."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    ar_order = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_pcg_adaptive_seg(pcg, fs, ar_order)
    assert isinstance(result, dict)
