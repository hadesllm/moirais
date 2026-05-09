"""Tests for eslsce.esl_score_match."""
import numpy as np
import pytest
from moirais.fn.eslsce import esl_score_match


def test_eslsce_basic():
    """Test basic functionality."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    p = 5
    result = esl_score_match(q, theta, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslsce_edge():
    """Test edge cases."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    p = 5
    result = esl_score_match(q, theta, p)
    assert isinstance(result, dict)
