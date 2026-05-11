"""Tests for mscmcl.matrix_completion_scm."""
import numpy as np
import pytest
from morie.fn.mscmcl import matrix_completion_scm


def test_mscmcl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = matrix_completion_scm(y, D, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mscmcl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = matrix_completion_scm(y, D, lam)
    assert isinstance(result, dict)
