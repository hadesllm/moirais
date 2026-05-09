"""Tests for gh_c4_10.ghosal_dp_polya_urn."""
import numpy as np
import pytest
from moirais.fn.gh_c4_10 import ghosal_dp_polya_urn


def test_gh_c4_10_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_polya_urn(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c4_10_edge():
    """Test edge cases."""
    result = ghosal_dp_polya_urn(np.array([42.0]))
    assert result['n'] == 1
