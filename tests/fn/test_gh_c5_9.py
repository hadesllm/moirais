"""Tests for gh_c5_9.ghosal_beta_ker."""
import numpy as np
import pytest
from moirais.fn.gh_c5_9 import ghosal_beta_ker


def test_gh_c5_9_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_beta_ker(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c5_9_edge():
    """Test edge cases."""
    result = ghosal_beta_ker(np.array([42.0]))
    assert result['n'] == 1
