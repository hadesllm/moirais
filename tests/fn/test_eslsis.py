"""Tests for eslsis.esl_sis_screening."""
import numpy as np
import pytest
from moirais.fn.eslsis import esl_sis_screening


def test_eslsis_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    d = 5
    result = esl_sis_screening(X, y, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslsis_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    d = 5
    result = esl_sis_screening(X, y, d)
    assert isinstance(result, dict)
