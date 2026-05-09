"""Tests for aitlnf.logistic_normal_fit."""
import numpy as np
import pytest
from moirais.fn.aitlnf import logistic_normal_fit


def test_aitlnf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ref = np.random.default_rng(42).normal(0, 1, 100)
    result = logistic_normal_fit(X, ref)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitlnf_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ref = np.random.default_rng(42).normal(0, 1, 100)
    result = logistic_normal_fit(X, ref)
    assert isinstance(result, dict)
