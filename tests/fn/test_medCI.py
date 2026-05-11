"""Tests for medCI.asymmetric_indirect_ci."""
import numpy as np
import pytest
from morie.fn.medCI import asymmetric_indirect_ci


def test_medCI_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    sa = np.random.default_rng(42).normal(0, 1, 100)
    sb = np.random.default_rng(42).normal(0, 1, 100)
    n_sim = np.random.default_rng(42).normal(0, 1, 100)
    result = asymmetric_indirect_ci(a, b, sa, sb, n_sim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_medCI_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    sa = np.random.default_rng(42).normal(0, 1, 100)
    sb = np.random.default_rng(42).normal(0, 1, 100)
    n_sim = np.random.default_rng(42).normal(0, 1, 100)
    result = asymmetric_indirect_ci(a, b, sa, sb, n_sim)
    assert isinstance(result, dict)
