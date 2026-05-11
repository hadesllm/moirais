"""Tests for eslboo.esl_bootstrap_err."""
import numpy as np
import pytest
from morie.fn.eslboo import esl_bootstrap_err


def test_eslboo_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = esl_bootstrap_err(X, y, model, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslboo_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = esl_bootstrap_err(X, y, model, B)
    assert isinstance(result, dict)
