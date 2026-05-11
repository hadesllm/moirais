"""Tests for ghsve.ghosal_sieve_prior."""
import numpy as np
import pytest
from morie.fn.ghsve import ghosal_sieve_prior


def test_ghsve_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_sieve_prior(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ghsve_edge():
    """Test edge cases."""
    result = ghosal_sieve_prior(np.array([42.0]))
    assert result['n'] == 1
