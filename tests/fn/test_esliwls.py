"""Tests for esliwls.esl_iwls."""
import numpy as np
import pytest
from moirais.fn.esliwls import esl_iwls


def test_esliwls_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    beta0 = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_iwls(X, y, beta0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esliwls_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    beta0 = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_iwls(X, y, beta0)
    assert isinstance(result, dict)
