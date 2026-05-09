"""Tests for gh_c3_16.ghosal_evsplit_pt."""
import numpy as np
import pytest
from moirais.fn.gh_c3_16 import ghosal_evsplit_pt


def test_gh_c3_16_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_evsplit_pt(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c3_16_edge():
    """Test edge cases."""
    result = ghosal_evsplit_pt(np.array([42.0]))
    assert result['n'] == 1
