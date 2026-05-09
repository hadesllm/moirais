"""Tests for rgpsd.rangayyan_psd."""
import numpy as np
import pytest
from moirais.fn.rgpsd import rangayyan_psd


def test_rgpsd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_psd(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgpsd_edge():
    """Test edge cases."""
    result = rangayyan_psd(np.array([42.0]))
    assert result['n'] == 1
