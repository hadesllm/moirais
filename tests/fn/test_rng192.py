"""Tests for rng192.rangayyan_ch4_pan_tompkins_searchback_update."""
import numpy as np
import pytest
from morie.fn.rng192 import rangayyan_ch4_pan_tompkins_searchback_update


def test_rng192_basic():
    """Test basic functionality."""
    PEAKI = np.random.default_rng(42).normal(0, 1, 100)
    SPKI = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_pan_tompkins_searchback_update(PEAKI, SPKI)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng192_edge():
    """Test edge cases."""
    PEAKI = np.random.default_rng(42).normal(0, 1, 100)
    SPKI = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_pan_tompkins_searchback_update(PEAKI, SPKI)
    assert isinstance(result, dict)
