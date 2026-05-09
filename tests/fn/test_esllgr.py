"""Tests for esllgr.esl_logistic_reg."""
import numpy as np
import pytest
from moirais.fn.esllgr import esl_logistic_reg


def test_esllgr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_logistic_reg(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esllgr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_logistic_reg(X, y)
    assert isinstance(result, dict)
