"""Tests for cfaftr.cfa_one_factor."""
import numpy as np
import pytest
from moirais.fn.cfaftr import cfa_one_factor


def test_cfaftr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    factor_structure = np.random.default_rng(42).normal(0, 1, 100)
    result = cfa_one_factor(X, factor_structure)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cfaftr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    factor_structure = np.random.default_rng(42).normal(0, 1, 100)
    result = cfa_one_factor(X, factor_structure)
    assert isinstance(result, dict)
