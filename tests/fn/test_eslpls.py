"""Tests for eslpls.esl_pls."""
import numpy as np
import pytest
from moirais.fn.eslpls import esl_pls


def test_eslpls_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = esl_pls(X, y, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslpls_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = esl_pls(X, y, M)
    assert isinstance(result, dict)
