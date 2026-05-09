"""Tests for esllpr.esl_local_linear."""
import numpy as np
import pytest
from moirais.fn.esllpr import esl_local_linear


def test_esllpr_basic():
    """Test basic functionality."""
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_local_linear(x0, x, y, lambda_)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esllpr_edge():
    """Test edge cases."""
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_local_linear(x0, x, y, lambda_)
    assert isinstance(result, dict)
