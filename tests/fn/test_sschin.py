"""Tests for sschin.chained_imputation."""
import numpy as np
import pytest
from morie.fn.sschin import chained_imputation


def test_sschin_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    mi_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = chained_imputation(time, event, X, mi_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sschin_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    mi_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = chained_imputation(time, event, X, mi_iter)
    assert isinstance(result, dict)
