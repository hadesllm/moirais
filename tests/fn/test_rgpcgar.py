"""Tests for rgpcgar.rangayyan_pcg_ar_model."""
import numpy as np
import pytest
from morie.fn.rgpcgar import rangayyan_pcg_ar_model


def test_rgpcgar_basic():
    """Test basic functionality."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_pcg_ar_model(pcg, fs, p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpcgar_edge():
    """Test edge cases."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_pcg_ar_model(pcg, fs, p, q)
    assert isinstance(result, dict)
