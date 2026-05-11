"""Tests for eslnmf.esl_nmf."""
import numpy as np
import pytest
from morie.fn.eslnmf import esl_nmf


def test_eslnmf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = esl_nmf(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslnmf_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = esl_nmf(X, k)
    assert isinstance(result, dict)
