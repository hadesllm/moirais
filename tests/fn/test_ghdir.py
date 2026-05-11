"""Tests for ghdir.ghosal_dirichlet_posterior."""
import numpy as np
import pytest
from morie.fn.ghdir import ghosal_dirichlet_posterior


def test_ghdir_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dirichlet_posterior(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ghdir_edge():
    """Test edge cases."""
    result = ghosal_dirichlet_posterior(np.array([42.0]))
    assert result['n'] == 1
