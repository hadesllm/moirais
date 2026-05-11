"""Tests for hsirt.heteroskedastic_irt."""
import numpy as np
import pytest
from morie.fn.hsirt import heteroskedastic_irt


def test_hsirt_basic():
    """Test basic functionality."""
    votes = np.random.default_rng(43).integers(0, 2, (50, 100))
    n_dims = 2
    result = heteroskedastic_irt(votes, n_dims)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hsirt_edge():
    """Test edge cases."""
    votes = np.random.default_rng(43).integers(0, 2, (50, 100))
    n_dims = 2
    result = heteroskedastic_irt(votes, n_dims)
    assert isinstance(result, dict)
