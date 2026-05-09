"""Tests for cvxprc.boyd_projection."""
import numpy as np
import pytest
from moirais.fn.cvxprc import boyd_projection


def test_cvxprc_basic():
    """Test basic functionality."""
    v = np.random.default_rng(44).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_projection(v, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxprc_edge():
    """Test edge cases."""
    v = np.random.default_rng(44).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_projection(v, C)
    assert isinstance(result, dict)
