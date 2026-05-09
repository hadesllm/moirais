"""Tests for strtwt.stratified_weights."""
import numpy as np
import pytest
from moirais.fn.strtwt import stratified_weights


def test_strtwt_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = stratified_weights(A, H, S)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_strtwt_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = stratified_weights(A, H, S)
    assert isinstance(result, dict)
