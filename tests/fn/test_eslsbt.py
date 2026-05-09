"""Tests for eslsbt.esl_se_beta."""
import numpy as np
import pytest
from moirais.fn.eslsbt import esl_se_beta


def test_eslsbt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    beta = 0.8
    result = esl_se_beta(X, y, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslsbt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    beta = 0.8
    result = esl_se_beta(X, y, beta)
    assert isinstance(result, dict)
