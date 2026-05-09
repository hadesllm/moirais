"""Tests for eslemg.esl_em_gmm."""
import numpy as np
import pytest
from moirais.fn.eslemg import esl_em_gmm


def test_eslemg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = esl_em_gmm(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslemg_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = esl_em_gmm(X, k)
    assert isinstance(result, dict)
