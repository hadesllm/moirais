"""Tests for rgcepst.rangayyan_cepstrum."""
import numpy as np
import pytest
from morie.fn.rgcepst import rangayyan_cepstrum


def test_rgcepst_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_cepstrum(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgcepst_edge():
    """Test edge cases."""
    result = rangayyan_cepstrum(np.array([42.0]))
    assert result['n'] == 1
