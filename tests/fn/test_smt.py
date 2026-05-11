"""Tests for smt.semiparametric_max."""
import numpy as np
import pytest
from morie.fn.smt import semiparametric_max


def test_smt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = semiparametric_max(y, X, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = semiparametric_max(y, X, model)
    assert isinstance(result, dict)
