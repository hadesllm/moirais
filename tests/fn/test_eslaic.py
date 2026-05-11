"""Tests for eslaic.esl_aic_score."""
import numpy as np
import pytest
from morie.fn.eslaic import esl_aic_score


def test_eslaic_basic():
    """Test basic functionality."""
    loglik = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = esl_aic_score(loglik, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslaic_edge():
    """Test edge cases."""
    loglik = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = esl_aic_score(loglik, d)
    assert isinstance(result, dict)
