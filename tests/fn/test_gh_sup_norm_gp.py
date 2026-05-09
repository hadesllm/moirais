"""Tests for gh_sup_norm_gp.ghosal_sup_norm_contraction."""
import numpy as np
import pytest
from moirais.fn.gh_sup_norm_gp import ghosal_sup_norm_contraction


def test_gh_sup_norm_gp_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_sup_norm_contraction(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_sup_norm_gp_edge():
    """Test edge cases."""
    result = ghosal_sup_norm_contraction(np.array([42.0]))
    assert result['n'] == 1
