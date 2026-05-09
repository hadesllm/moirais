"""Tests for cvxell.boyd_minvol_ellipsoid."""
import numpy as np
import pytest
from moirais.fn.cvxell import boyd_minvol_ellipsoid


def test_cvxell_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = boyd_minvol_ellipsoid(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxell_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = boyd_minvol_ellipsoid(X)
    assert isinstance(result, dict)
