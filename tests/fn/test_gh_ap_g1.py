"""Tests for gh_ap_g1.ghosal_fin_dir_def."""
import numpy as np
import pytest
from moirais.fn.gh_ap_g1 import ghosal_fin_dir_def


def test_gh_ap_g1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_fin_dir_def(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_ap_g1_edge():
    """Test edge cases."""
    result = ghosal_fin_dir_def(np.array([42.0]))
    assert result['n'] == 1
