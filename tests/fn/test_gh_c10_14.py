"""Tests for gh_c10_14.ghosal_param_np_bf."""
import numpy as np
import pytest
from moirais.fn.gh_c10_14 import ghosal_param_np_bf


def test_gh_c10_14_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_param_np_bf(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c10_14_edge():
    """Test edge cases."""
    result = ghosal_param_np_bf(np.array([42.0]))
    assert result['n'] == 1
