"""Tests for aitprm.compositional_permanova."""
import numpy as np
import pytest
from morie.fn.aitprm import compositional_permanova


def test_aitprm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    groups = np.random.default_rng(43).integers(0, 3, 100)
    n_perm = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_permanova(X, groups, n_perm)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitprm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    groups = np.random.default_rng(43).integers(0, 3, 100)
    n_perm = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_permanova(X, groups, n_perm)
    assert isinstance(result, dict)
