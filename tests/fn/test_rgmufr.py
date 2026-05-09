"""Tests for rgmufr.rangayyan_muap_firing_rate."""
import numpy as np
import pytest
from moirais.fn.rgmufr import rangayyan_muap_firing_rate


def test_rgmufr_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_muap_firing_rate(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgmufr_edge():
    """Test edge cases."""
    result = rangayyan_muap_firing_rate(np.array([42.0]))
    assert result['n'] == 1
