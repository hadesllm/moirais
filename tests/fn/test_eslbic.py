"""Tests for eslbic.esl_bic_score."""
import numpy as np
import pytest
from moirais.fn.eslbic import esl_bic_score


def test_eslbic_basic():
    """Test basic functionality."""
    loglik = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    N = 100
    result = esl_bic_score(loglik, d, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslbic_edge():
    """Test edge cases."""
    loglik = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    N = 100
    result = esl_bic_score(loglik, d, N)
    assert isinstance(result, dict)
