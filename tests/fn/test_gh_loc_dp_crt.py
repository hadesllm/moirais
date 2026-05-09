"""Tests for gh_loc_dp_crt.ghosal_local_dp_rate."""
import numpy as np
import pytest
from moirais.fn.gh_loc_dp_crt import ghosal_local_dp_rate


def test_gh_loc_dp_crt_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_local_dp_rate(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_loc_dp_crt_edge():
    """Test edge cases."""
    result = ghosal_local_dp_rate(np.array([42.0]))
    assert result['n'] == 1
