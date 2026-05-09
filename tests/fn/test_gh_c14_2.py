"""Tests for gh_c14_2.ghosal_ewens_esf."""
import numpy as np
import pytest
from moirais.fn.gh_c14_2 import ghosal_ewens_esf


def test_gh_c14_2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_ewens_esf(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c14_2_edge():
    """Test edge cases."""
    result = ghosal_ewens_esf(np.array([42.0]))
    assert result['n'] == 1
