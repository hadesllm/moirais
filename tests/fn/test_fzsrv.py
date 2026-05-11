"""Tests for fzsrv.fauzi_survival_kernel."""
import numpy as np
import pytest
from morie.fn.fzsrv import fauzi_survival_kernel


def test_fzsrv_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_survival_kernel(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_fzsrv_edge():
    """Test edge cases."""
    result = fauzi_survival_kernel(np.array([42.0]))
    assert result['n'] == 1
