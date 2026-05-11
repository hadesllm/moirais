"""Tests for rgam.rangayyan_am_signal."""
import numpy as np
import pytest
from morie.fn.rgam import rangayyan_am_signal


def test_rgam_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    fc = np.random.default_rng(42).normal(0, 1, 100)
    m_t = np.random.default_rng(42).normal(0, 1, 100)
    Ac = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_am_signal(t, fc, m_t, Ac)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgam_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    fc = np.random.default_rng(42).normal(0, 1, 100)
    m_t = np.random.default_rng(42).normal(0, 1, 100)
    Ac = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_am_signal(t, fc, m_t, Ac)
    assert isinstance(result, dict)
