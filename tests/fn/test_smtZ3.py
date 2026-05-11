"""Tests for smtZ3.smt_solver."""
import numpy as np
import pytest
from morie.fn.smtZ3 import smt_solver


def test_smtZ3_basic():
    """Test basic functionality."""
    formula = np.random.default_rng(42).normal(0, 1, 100)
    result = smt_solver(formula)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smtZ3_edge():
    """Test edge cases."""
    formula = np.random.default_rng(42).normal(0, 1, 100)
    result = smt_solver(formula)
    assert isinstance(result, dict)
