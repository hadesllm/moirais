"""Tests for esldat.esl_dropout."""
import numpy as np
import pytest
from moirais.fn.esldat import esl_dropout


def test_esldat_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    p = 5
    result = esl_dropout(X, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esldat_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    p = 5
    result = esl_dropout(X, p)
    assert isinstance(result, dict)
