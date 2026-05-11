"""Tests for gh_ap_a1.ghosal_weak_conv_def."""
import numpy as np
import pytest
from morie.fn.gh_ap_a1 import ghosal_weak_conv_def


def test_gh_ap_a1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_weak_conv_def(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_ap_a1_edge():
    """Test edge cases."""
    result = ghosal_weak_conv_def(np.array([42.0]))
    assert result['n'] == 1
