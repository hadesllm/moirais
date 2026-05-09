"""Tests for eslprt.esl_partial_dependence."""
import numpy as np
import pytest
from moirais.fn.eslprt import esl_partial_dependence


def test_eslprt_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_partial_dependence(model, X, S)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslprt_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_partial_dependence(model, X, S)
    assert isinstance(result, dict)
