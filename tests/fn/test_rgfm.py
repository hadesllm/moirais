"""Tests for rgfm.rangayyan_fm_signal."""
import numpy as np
import pytest
from moirais.fn.rgfm import rangayyan_fm_signal


def test_rgfm_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    f0 = np.random.default_rng(42).normal(0, 1, 100)
    m_t = np.random.default_rng(42).normal(0, 1, 100)
    kf = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_fm_signal(t, f0, m_t, kf)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgfm_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    f0 = np.random.default_rng(42).normal(0, 1, 100)
    m_t = np.random.default_rng(42).normal(0, 1, 100)
    kf = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_fm_signal(t, f0, m_t, kf)
    assert isinstance(result, dict)
