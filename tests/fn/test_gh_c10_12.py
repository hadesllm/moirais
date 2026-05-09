"""Tests for gh_c10_12.ghosal_modsel_bic."""
import numpy as np
import pytest
from moirais.fn.gh_c10_12 import ghosal_modsel_bic


def test_gh_c10_12_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_modsel_bic(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c10_12_edge():
    """Test edge cases."""
    result = ghosal_modsel_bic(np.array([42.0]))
    assert result['n'] == 1
