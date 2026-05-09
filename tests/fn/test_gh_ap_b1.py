"""Tests for gh_ap_b1.ghosal_kl_props."""
import numpy as np
import pytest
from moirais.fn.gh_ap_b1 import ghosal_kl_props


def test_gh_ap_b1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_kl_props(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_ap_b1_edge():
    """Test edge cases."""
    result = ghosal_kl_props(np.array([42.0]))
    assert result['n'] == 1
