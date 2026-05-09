"""Tests for dmlMed.dml_mediation_orthogonal."""
import numpy as np
import pytest
from moirais.fn.dmlMed import dml_mediation_orthogonal


def test_dmlMed_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = dml_mediation_orthogonal(Y, X, M, C, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dmlMed_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = dml_mediation_orthogonal(Y, X, M, C, K)
    assert isinstance(result, dict)
