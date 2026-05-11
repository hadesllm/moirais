"""Tests for rptpn.repetition_penalty."""
import numpy as np
import pytest
from morie.fn.rptpn import repetition_penalty


def test_rptpn_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = repetition_penalty(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rptpn_edge():
    """Test edge cases."""
    result = repetition_penalty(np.array([42.0]))
    assert result['n'] == 1
