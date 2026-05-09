"""Tests for cvxslt.boyd_slater."""
import numpy as np
import pytest
from moirais.fn.cvxslt import boyd_slater


def test_cvxslt_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_slater(f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxslt_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_slater(f)
    assert isinstance(result, dict)
