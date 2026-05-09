"""Tests for gh_c2_6.ghosal_mixture_basis_prior."""
import numpy as np
import pytest
from moirais.fn.gh_c2_6 import ghosal_mixture_basis_prior


def test_gh_c2_6_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_mixture_basis_prior(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_gh_c2_6_edge():
    """Test edge cases."""
    result = ghosal_mixture_basis_prior(np.array([42.0]))
    assert result['n'] == 1
