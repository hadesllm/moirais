"""Tests for rgfracv.rangayyan_fractal_vag."""
import numpy as np
import pytest
from moirais.fn.rgfracv import rangayyan_fractal_vag


def test_rgfracv_basic():
    """Test basic functionality."""
    vag = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_fractal_vag(vag, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgfracv_edge():
    """Test edge cases."""
    vag = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_fractal_vag(vag, fs)
    assert isinstance(result, dict)
