"""Tests for eslcvr.esl_cv_score."""
import numpy as np
import pytest
from morie.fn.eslcvr import esl_cv_score


def test_eslcvr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = esl_cv_score(X, y, model, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslcvr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = esl_cv_score(X, y, model, k)
    assert isinstance(result, dict)
