"""Tests for comple.complete_case."""
import numpy as np
import pytest
from moirais.fn.comple import complete_case


def test_comple_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = complete_case(y, X, R)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_comple_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = complete_case(y, X, R)
    assert isinstance(result, dict)
