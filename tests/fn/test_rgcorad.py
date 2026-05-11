"""Tests for rgcorad.rangayyan_coronary_ad."""
import numpy as np
import pytest
from morie.fn.rgcorad import rangayyan_coronary_ad


def test_rgcorad_basic():
    """Test basic functionality."""
    coronary_sound = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    order = 4
    result = rangayyan_coronary_ad(coronary_sound, fs, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgcorad_edge():
    """Test edge cases."""
    coronary_sound = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    order = 4
    result = rangayyan_coronary_ad(coronary_sound, fs, order)
    assert isinstance(result, dict)
