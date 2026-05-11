"""Tests for gh_c13_2.ghosal_surv_dp_km."""
import numpy as np
import pytest
from morie.fn.gh_c13_2 import ghosal_surv_dp_km


def test_gh_c13_2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_surv_dp_km(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c13_2_edge():
    """Test edge cases."""
    result = ghosal_surv_dp_km(np.array([42.0]))
    assert result['n'] == 1
