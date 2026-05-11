"""Tests for gh_c8_7.ghosal_fin_apx_pri."""
import numpy as np
import pytest
from morie.fn.gh_c8_7 import ghosal_fin_apx_pri


def test_gh_c8_7_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_fin_apx_pri(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c8_7_edge():
    """Test edge cases."""
    result = ghosal_fin_apx_pri(np.array([42.0]))
    assert result['n'] == 1
