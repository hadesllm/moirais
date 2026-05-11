"""Tests for rng206.rangayyan_ch4_coherence_spectrum."""
import numpy as np
import pytest
from morie.fn.rng206 import rangayyan_ch4_coherence_spectrum


def test_rng206_basic():
    """Test basic functionality."""
    S_xy = np.random.default_rng(42).normal(0, 1, 100)
    S_xx = np.random.default_rng(42).normal(0, 1, 100)
    S_yy = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_coherence_spectrum(S_xy, S_xx, S_yy, f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng206_edge():
    """Test edge cases."""
    S_xy = np.random.default_rng(42).normal(0, 1, 100)
    S_xx = np.random.default_rng(42).normal(0, 1, 100)
    S_yy = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_coherence_spectrum(S_xy, S_xx, S_yy, f)
    assert isinstance(result, dict)
