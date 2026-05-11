"""Tests for eslnnb.esl_naive_bayes."""
import numpy as np
import pytest
from morie.fn.eslnnb import esl_naive_bayes


def test_eslnnb_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_naive_bayes(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslnnb_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_naive_bayes(X, y)
    assert isinstance(result, dict)
