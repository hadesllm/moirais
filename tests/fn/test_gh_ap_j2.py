"""Tests for gh_ap_j2.ghosal_crm_laplace."""
import numpy as np
import pytest
from moirais.fn.gh_ap_j2 import ghosal_crm_laplace


def test_gh_ap_j2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_crm_laplace(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_ap_j2_edge():
    """Test edge cases."""
    result = ghosal_crm_laplace(np.array([42.0]))
    assert result['n'] == 1
