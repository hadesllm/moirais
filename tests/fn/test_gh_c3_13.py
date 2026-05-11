"""Tests for gh_c3_13.ghosal_polya_urn_pt."""
import numpy as np
import pytest
from morie.fn.gh_c3_13 import ghosal_polya_urn_pt


def test_gh_c3_13_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_polya_urn_pt(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c3_13_edge():
    """Test edge cases."""
    result = ghosal_polya_urn_pt(np.array([42.0]))
    assert result['n'] == 1
