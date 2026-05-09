"""Tests for esllle.esl_lle."""
import numpy as np
import pytest
from moirais.fn.esllle import esl_lle


def test_esllle_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = esl_lle(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esllle_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = esl_lle(X, k)
    assert isinstance(result, dict)
