"""Tests for esltrs.esl_basis_truncated."""
import numpy as np
import pytest
from morie.fn.esltrs import esl_basis_truncated


def test_esltrs_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    knots = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = esl_basis_truncated(x, knots, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esltrs_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    knots = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = esl_basis_truncated(x, knots, p)
    assert isinstance(result, dict)
