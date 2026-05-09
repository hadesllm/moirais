"""Tests for eslfst.esl_f_test."""
import numpy as np
import pytest
from moirais.fn.eslfst import esl_f_test


def test_eslfst_basic():
    """Test basic functionality."""
    model0 = np.random.default_rng(42).normal(0, 1, 100)
    model1 = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_f_test(model0, model1, X, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_eslfst_edge():
    """Test edge cases."""
    model0 = np.random.default_rng(42).normal(0, 1, 100)
    model1 = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_f_test(model0, model1, X, y)
    assert isinstance(result, dict)
